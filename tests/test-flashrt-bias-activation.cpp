// SPDX-License-Identifier: MIT
// Host contracts and CUDA graph fixtures for the SM89 F32 bias + GELU/SiLU
// adapter. GPU results are compared bitwise across independent mode processes
// by the runner. CPU transcendental functions provide a tolerance reference,
// never a binary oracle. Alias mutations below are host-only contract checks.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-cuda/flashrt-bias-activation-guard.h"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
static void need(bool ok, const std::string & message) {
    if (!ok) { throw std::runtime_error(message); }
}
static void env(const char * key, const char * value) {
#if defined(_WIN32)
    need(_putenv_s(key, value ? value : "") == 0, key);
#else
    need((value ? setenv(key, value, 1) : unsetenv(key)) == 0, key);
#endif
}
static std::string quoted(const std::string & value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char ch : value) {
        if (ch == '"' || ch == '\\') { out << '\\' << char(ch); }
        else if (ch < 32) { out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(ch) << std::dec; }
        else { out << char(ch); }
    }
    out << '"';
    return out.str();
}
struct options {
    bool host_only = false;
    bool graphs = false;
    int mode = -1;
    std::string prefix;
};
static options parse(int argc, char ** argv) {
    options opts;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--host-only") { opts.host_only = true; continue; }
        if (key == "--help") {
            std::cout << "Usage: test-flashrt-bias-activation --host-only\n"
                         "   or: --mode 0|1 --graphs off|on --output-prefix PATH\n";
            std::exit(0);
        }
        need(i + 1 < argc, "missing value after " + key);
        const std::string value = argv[++i];
        if (key == "--mode") {
            need(value == "0" || value == "1", "mode must be 0 or 1");
            opts.mode = value == "1" ? 1 : 0;
        } else if (key == "--graphs") {
            need(value == "off" || value == "on", "graphs must be off or on");
            opts.graphs = value == "on";
        } else if (key == "--output-prefix") { opts.prefix = value; }
        else { throw std::runtime_error("unknown option: " + key); }
    }
    need(opts.host_only || (opts.mode >= 0 && !opts.prefix.empty()), "mode and output-prefix are required");
    return opts;
}

struct case_spec {
    std::string name;
    std::string kind = "gelu";
    int width = 4304;
    int rows = 50;
    bool cont = false;
    bool bias_first = false;
    std::string variant = "normal";
    std::string data = "random";
    bool eligible = true;
};
static case_spec spec_for(const std::string & kind, bool cont, int rows, const std::string & suffix) {
    case_spec spec;
    spec.name = kind + (cont ? "_cont_" : "_direct_") + suffix;
    spec.kind = kind;
    spec.width = kind == "gelu" ? 4304 : 1024;
    spec.rows = rows;
    spec.cont = cont;
    return spec;
}
static std::vector<case_spec> gpu_cases() {
    std::vector<case_spec> cases;
    for (const std::string kind : {"gelu", "silu"}) {
        const std::vector<int> shapes = kind == "gelu" ? std::vector<int>{1, 50, 256, 512}
                                                       : std::vector<int>{1, 10, 50};
        for (bool cont : {false, true}) {
            for (int rows : shapes) { cases.push_back(spec_for(kind, cont, rows, "r" + std::to_string(rows))); }
            case_spec first = spec_for(kind, cont, 1, "bias-first");
            first.bias_first = true;
            cases.push_back(first);
        }
        for (const std::string data : {"cancellation", "saturated", "signed-zero"}) {
            case_spec spec = spec_for(kind, false, 50, data);
            spec.data = data;
            cases.push_back(spec);
        }
        case_spec zero = spec_for(kind, true, 50, "signed-zero");
        zero.data = "signed-zero";
        cases.push_back(zero);
        int variant_index = 0;
        for (const std::string variant : {"wrong-unary", "wrong-width", "wrong-rows", "wrong-bias-shape",
                                          "strided-x", "wrong-edge", "intermediate-output", "extra-consumer"}) {
            case_spec spec = spec_for(kind, (variant_index++ % 2) != 0, variant == "wrong-rows" ? 7 : 50,
                                      "fallback_" + variant);
            spec.variant = variant;
            spec.eligible = false;
            if (variant == "wrong-width") { spec.width = kind == "gelu" ? 4096 : 2048; }
            cases.push_back(spec);
        }
        case_spec kept = spec_for(kind, true, 50, "fallback_unary-output");
        kept.variant = "unary-output";
        kept.eligible = false;
        cases.push_back(kept);
    }
    return cases;
}

struct fixture {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * x = nullptr;
    ggml_tensor * x_storage = nullptr;
    ggml_tensor * bias = nullptr;
    ggml_tensor * add = nullptr;
    ggml_tensor * unary = nullptr;
    ggml_tensor * cont = nullptr;
    ggml_tensor * out = nullptr;
    ggml_tensor * final = nullptr;
    ggml_cgraph * graph = nullptr;
    int index = -1;
    case_spec spec;

    explicit fixture(case_spec s, bool host = true) : spec(std::move(s)) {
        ctx = ggml_init({host ? size_t(64 * 1024 * 1024) : size_t(256 * 1024), nullptr, !host});
        need(ctx != nullptr, "GGML metadata allocation failed");
        const bool strided = spec.variant == "strided-x";
        x_storage = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, spec.width * (strided ? 2 : 1), spec.rows);
        x = strided ? ggml_view_2d(ctx, x_storage, spec.width, spec.rows, x_storage->nb[1], 0) : x_storage;
        bias = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, spec.width,
                                 spec.variant == "wrong-bias-shape" ? spec.rows : 1);
        add = spec.bias_first ? ggml_add(ctx, bias, x) : ggml_add(ctx, x, bias);
        ggml_tensor * input = spec.variant == "wrong-edge" ? x : add;
        unary = spec.variant == "wrong-unary" ? ggml_abs(ctx, input)
              : spec.kind == "gelu" ? ggml_gelu(ctx, input) : ggml_silu(ctx, input);
        if (spec.cont) { cont = ggml_cont(ctx, unary); }
        out = cont ? cont : unary;
        final = spec.variant == "extra-consumer" ? ggml_add(ctx, out, add) : out;
        ggml_set_name(x, ("activation_test_" + spec.name + "_x").c_str());
        ggml_set_name(bias, (spec.name + "_bias").c_str());
        ggml_set_name(add, ("activation_test_" + spec.name + "_add").c_str());
        ggml_set_name(unary, (spec.name + "_unary").c_str());
        ggml_set_name(out, ("activation_test_" + spec.name + "_out").c_str());
        if (final != out) { ggml_set_name(final, (spec.name + "_final").c_str()); }
        ggml_set_input(x_storage);
        ggml_set_input(bias);
        ggml_set_output(final);
        if (spec.variant == "intermediate-output") { ggml_set_output(add); }
        if (spec.variant == "unary-output") { ggml_set_output(unary); }
        graph = ggml_new_graph_custom(ctx, 32, false);
        if (spec.variant == "wrong-edge") { ggml_build_forward_expand(graph, add); }
        ggml_build_forward_expand(graph, final);
        ggml_set_name(graph->nodes[0], ("activation_test_" + spec.name + "_first").c_str());
        for (int i = 0; i < graph->n_nodes; ++i) {
            if (graph->nodes[i] == add) { index = i; break; }
        }
        need(index >= 0, "initial ADD missing from graph");
    }
    fixture(const fixture &) = delete;
    fixture & operator=(const fixture &) = delete;
    ~fixture() {
        if (buffer) { ggml_backend_buffer_free(buffer); }
        if (ctx) { ggml_free(ctx); }
    }
    bool accepts() const {
        ggml_flashrt_bias_activation_plan plan{};
        return ggml_flashrt_plan_bias_activation(graph, index, plan) == (spec.cont ? 2 : 1);
    }
    void allocate(ggml_backend_t backend) {
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        need(buffer != nullptr, "CUDA tensor allocation failed");
    }
};

static int host_checks() {
    int checks = 0;
    auto check = [&](bool actual, bool expected, const std::string & label) {
        ++checks;
        need(actual == expected, "host contract: " + label);
    };
    for (const case_spec & spec : gpu_cases()) {
        fixture f(spec);
        check(f.accepts(), spec.eligible, spec.name);
        if (spec.eligible) {
            ggml_flashrt_bias_activation_plan plan{};
            (void)ggml_flashrt_plan_bias_activation(f.graph, f.index, plan);
            // One-row bias-first graphs have indistinguishable input layouts;
            // the plan must preserve src0/src1 order without guessing roles.
            check(plan.x == f.add->src[0] && plan.bias == f.add->src[1] && plan.add == f.add && plan.unary == f.unary &&
                  plan.cont == f.cont && plan.output == f.out && plan.width == spec.width && plan.rows == spec.rows &&
                  !plan.bias_first && plan.kind == (spec.kind == "gelu" ? GGML_UNARY_OP_GELU : GGML_UNARY_OP_SILU),
                  true, "plan fields: " + spec.name);
        }
    }
    for (bool cont : {false, true}) {
        const case_spec spec = spec_for("gelu", cont, 50, "host");
        { fixture f(spec); f.out->data = f.x->data;
          check(f.accepts(), true, "output exactly aliases x"); }
        { fixture f(spec); f.add->data = f.x->data; f.x->flags &= ~GGML_TENSOR_FLAG_INPUT;
          check(f.accepts(), true, "ADD exactly aliases unobservable x"); }
        { fixture f(spec); f.add->data = f.x->data;
          check(f.accepts(), false, "omitted ADD write to observable INPUT x"); }
        { fixture f(spec); f.add->data = f.x->data; f.out->data = f.x->data;
          check(f.accepts(), true, "final output writes aliased INPUT x"); }
        { fixture f(spec); f.add->data = f.x->data; f.x->flags &= ~GGML_TENSOR_FLAG_INPUT;
          ggml_build_forward_expand(f.graph, ggml_add(f.ctx, f.final, f.x));
          check(f.accepts(), false, "future x read sees omitted ADD write"); }
        { fixture f(spec); f.out->data = static_cast<char *>(f.x->data) + 8;
          check(f.accepts(), false, "partially overlapping output x"); }
        { fixture f(spec); f.add->data = static_cast<char *>(f.x->data) + 8;
          check(f.accepts(), false, "partially overlapping ADD x"); }
        { fixture f(spec); f.out->data = f.bias->data;
          check(f.accepts(), false, "output writes shared bias"); }
        { fixture f(spec); f.add->data = f.bias->data;
          check(f.accepts(), false, "ADD writes shared bias"); }
        { fixture f(spec); f.x->data = static_cast<char *>(f.x->data) + 4;
          check(f.accepts(), false, "misaligned x"); }
        { fixture f(spec); f.bias->data = static_cast<char *>(f.bias->data) + 4;
          check(f.accepts(), false, "misaligned bias"); }
        { fixture f(spec); f.out->data = nullptr;
          check(f.accepts(), false, "unallocated output"); }
        { fixture f(spec); f.bias->type = GGML_TYPE_F16;
          check(f.accepts(), false, "wrong bias dtype"); }
        { fixture f(spec); f.add->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
          check(f.accepts(), false, "ADD not computed"); }
        { fixture f(spec); f.unary->src[1] = f.bias;
          check(f.accepts(), false, "unary unexpected extra source"); }
        { fixture f(spec); f.add->view_src = f.x;
          check(f.accepts(), false, "ADD output view"); }
        { fixture f(spec); f.unary->view_src = f.add;
          check(f.accepts(), false, "UNARY output view"); }
        { fixture f(spec); ggml_tensor * view = ggml_view_tensor(f.ctx, f.add);
          ggml_build_forward_expand(f.graph, ggml_add(f.ctx, f.out, view));
          check(f.accepts(), false, "external raw ADD view without source edge"); }
        { fixture f(spec); ggml_flashrt_bias_activation_plan plan{};
          check(ggml_flashrt_plan_bias_activation(f.graph, -1, plan) != 0, false, "negative graph index");
          check(ggml_flashrt_plan_bias_activation(f.graph, f.graph->n_nodes, plan) != 0, false, "past-end graph index"); }
    }
    const case_spec cont_spec = spec_for("gelu", true, 50, "host-cont");
    { fixture f(cont_spec); f.cont->data = f.unary->data;
      check(f.accepts(), true, "CONT exactly aliases UNARY"); }
    { fixture f(cont_spec); f.unary->data = f.add->data;
      check(f.accepts(), true, "UNARY exactly aliases ADD"); }
    { fixture f(cont_spec); f.unary->data = f.x->data;
      check(f.accepts(), false, "omitted UNARY write to INPUT x before CONT"); }
    { fixture f(cont_spec); f.unary->data = f.x->data; f.x->flags &= ~GGML_TENSOR_FLAG_INPUT;
      ggml_build_forward_expand(f.graph, ggml_add(f.ctx, f.out, f.x));
      check(f.accepts(), false, "future x read sees omitted UNARY write before CONT"); }
    { fixture f(cont_spec); f.unary->data = f.x->data; f.x->flags &= ~GGML_TENSOR_FLAG_INPUT;
      check(f.accepts(), true, "UNARY exactly aliases unobservable x before CONT"); }
    { fixture f(cont_spec); f.add->data = f.x->data; f.unary->data = f.x->data; f.out->data = f.x->data;
      check(f.accepts(), true, "all outputs exactly alias x"); }
    { fixture f(cont_spec); f.cont->data = static_cast<char *>(f.unary->data) + 8;
      check(f.accepts(), false, "CONT partially overlaps UNARY"); }
    { fixture f(cont_spec); f.cont->nb[1] += 8;
      check(f.accepts(), false, "CONT different layout"); }
    { fixture f(cont_spec); f.cont->view_src = f.unary;
      check(f.accepts(), false, "CONT output view"); }
    { fixture f(cont_spec); ggml_tensor * view = ggml_view_tensor(f.ctx, f.unary);
      ggml_build_forward_expand(f.graph, ggml_add(f.ctx, f.out, view));
      check(f.accepts(), false, "external raw UNARY view"); }
    { fixture f(cont_spec); ggml_build_forward_expand(f.graph, ggml_add(f.ctx, f.out, f.unary));
      check(f.accepts(), false, "external UNARY consumer"); }
    { fixture f(cont_spec); check((f.cont->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 && f.accepts(), true,
                                  "final CONT OUTPUT preserved"); }
    std::cout << checks << " bias/activation contract checks passed; alias cases are host-only\n";
    return checks;
}

static float add_f32(float a, float b) { volatile float sum = a + b; return sum; }
static uint32_t bits(float value) { uint32_t out; std::memcpy(&out, &value, sizeof(out)); return out; }
struct input_values { std::vector<float> x, bias; };
static input_values make_inputs(const case_spec & spec) {
    input_values values;
    const size_t count = size_t(spec.width) * spec.rows;
    values.x.resize(count);
    values.bias.resize(spec.variant == "wrong-bias-shape" ? count : size_t(spec.width));
    uint32_t state = 0x64153791u;
    auto random = [&]() {
        state ^= state << 13; state ^= state >> 17; state ^= state << 5;
        return float(int(state & 0xfffffu) - 524288) / 262144.0f;
    };
    const float extremes[] = {-100.0f, -40.0f, -12.0f, -8.0f, -5.0f, -3.0f, -1.0f, -0.0f,
                              0.0f, 0.000001f, 1.0f, 3.0f, 5.0f, 8.0f, 12.0f, 40.0f, 100.0f};
    for (size_t i = 0; i < values.bias.size(); ++i) {
        values.bias[i] = spec.data == "cancellation" ? ((i & 1) ? -1048576.0f : 1048576.0f)
                       : spec.data == "signed-zero" ? ((i & 1) ? -0.0f : 0.0f)
                       : spec.data == "saturated" ? ((i & 1) ? -0.25f : 0.25f) : random() * 0.25f;
    }
    for (size_t i = 0; i < count; ++i) {
        const float b = values.bias[i % values.bias.size()];
        if (spec.data == "cancellation") {
            values.x[i] = -b + float(int(i % 49) - 24) * 0.125f;
        } else if (spec.data == "signed-zero") {
            values.x[i] = (i & 2) ? -0.0f : 0.0f;
        } else if (spec.data == "saturated") {
            values.x[i] = extremes[i % (sizeof(extremes) / sizeof(extremes[0]))] - b;
        } else { values.x[i] = random() * ((i % 5) ? 1.0f : 8.0f); }
    }
    return values;
}
static double activation_reference(const case_spec & spec, float input) {
    const double x = input;
    if (spec.variant == "wrong-unary") { return std::abs(x); }
    if (spec.kind == "gelu") {
        const double c = std::sqrt(2.0 / std::acos(-1.0));
        return 0.5 * x * (1.0 + std::tanh(c * x * (1.0 + 0.044715 * x * x)));
    }
    // Algebraically the same SiLU, with a stable sigmoid at both extremes.
    const double sigmoid = x >= 0 ? 1.0 / (1.0 + std::exp(-x)) : std::exp(x) / (1.0 + std::exp(x));
    return x * sigmoid;
}
static std::vector<double> reference(const case_spec & spec, const input_values & values,
                                      const std::string & tensor = "final") {
    std::vector<double> out(values.x.size());
    for (size_t i = 0; i < out.size(); ++i) {
        const float b = values.bias[i % values.bias.size()];
        const float a = spec.bias_first ? add_f32(b, values.x[i]) : add_f32(values.x[i], b);
        if (tensor == "add") { out[i] = a; continue; }
        out[i] = activation_reference(spec, spec.variant == "wrong-edge" ? values.x[i] : a);
        if (tensor == "final" && spec.variant == "extra-consumer") { out[i] = add_f32(float(out[i]), a); }
    }
    return out;
}
static void upload(fixture & f, const input_values & values) {
    if (f.spec.variant == "strided-x") {
        const size_t width = size_t(f.spec.width);
        std::vector<float> padded(width * 2 * f.spec.rows, 123.0f);
        for (int row = 0; row < f.spec.rows; ++row) {
            std::memcpy(padded.data() + size_t(row) * width * 2,
                        values.x.data() + size_t(row) * width, width * sizeof(float));
        }
        ggml_backend_tensor_set(f.x_storage, padded.data(), 0, padded.size() * sizeof(float));
    } else { ggml_backend_tensor_set(f.x, values.x.data(), 0, values.x.size() * sizeof(float)); }
    ggml_backend_tensor_set(f.bias, values.bias.data(), 0, values.bias.size() * sizeof(float));
}
struct reference_error { double maximum = 0.0; double rmse = 0.0; };
static reference_error compare_reference(const std::vector<double> & expected, const std::vector<float> & actual,
                                          const std::string & label) {
    need(expected.size() == actual.size() && !actual.empty(), "reference shape mismatch");
    reference_error error;
    double squares = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        need(std::isfinite(actual[i]) && std::isfinite(expected[i]), label + ": nonfinite output at " + std::to_string(i));
        const double difference = std::abs(double(actual[i]) - expected[i]);
        // This independent high-precision formula detects wrong activation,
        // missing bias, and reassociation across ADD. Exact CUDA equivalence is
        // a separate cross-process raw-byte comparison, including signed zero.
        const double bound = 2e-5 + 2e-5 * std::abs(expected[i]);
        if (difference > bound) {
            std::ostringstream message;
            message << label << ": independent reference mismatch at " << i
                    << " expected=" << std::setprecision(17) << expected[i] << " actual=" << actual[i]
                    << " abs_error=" << difference << " bound=" << bound;
            throw std::runtime_error(message.str());
        }
        error.maximum = std::max(error.maximum, difference);
        squares += difference * difference;
    }
    error.rmse = std::sqrt(squares / double(actual.size()));
    return error;
}
static void compare_repeat(const std::vector<float> & expected, const std::vector<float> & actual,
                             const std::string & label) {
    need(expected.size() == actual.size(), "repeat shape mismatch");
    for (size_t i = 0; i < actual.size(); ++i) {
        need(bits(expected[i]) == bits(actual[i]), label + ": repeated output bit mismatch at " + std::to_string(i));
    }
}
struct backend_owner {
    ggml_backend_t backend = nullptr;
    ~backend_owner() { if (backend) { ggml_backend_synchronize(backend); ggml_backend_free(backend); } }
};
static void gpu_tests(const options & opts, int checks) {
    env("GGML_CUDA_FLASHRT", "1");
    env("GGML_CUDA_FLASHRT_BIAS_ACTIVATION", opts.mode ? "1" : "0");
    env("GGML_CUDA_FLASHRT_TRACE", "1");
    env("GGML_CUDA_FLASHRT_ACTIVATION_GRAPH_TRACE", "1");
    env("GGML_CUDA_FLASHRT_SHADOW", "0");
    env("GGML_CUDA_DISABLE_FUSION", "0");
    env("GGML_CUDA_DISABLE_GRAPHS", opts.graphs ? nullptr : "1");
    backend_owner device;
    device.backend = ggml_backend_cuda_init(0);
    need(device.backend != nullptr, "CUDA0 backend initialization failed");
    const uint16_t endian = 1;
    need(*reinterpret_cast<const unsigned char *>(&endian) == 1, "raw export requires little-endian host");
    std::ofstream raw(opts.prefix + ".bin", std::ios::binary | std::ios::trunc);
    std::ofstream json(opts.prefix + ".json", std::ios::trunc);
    need(raw && json, "cannot open output-prefix files");
    json << std::setprecision(17);
    json << "{\"schema_version\":1,\"status\":\"tests_complete\",\"backend\":\"CUDA0\",\"mode\":" << opts.mode
         << ",\"graphs_requested\":" << (opts.graphs ? "true" : "false")
         << ",\"host_checks\":" << checks << ",\"case_count\":" << gpu_cases().size()
         << ",\"raw_dtype\":\"little-endian-f32\",\"raw_file\":" << quoted(opts.prefix + ".bin")
         << ",\"gpu_alias_cases\":0,\"reference\":\"sequential rounded F32 ADD followed by independent double GELU tanh approximation or SiLU sigmoid\""
         << ",\"reference_atol\":0.00002,\"reference_rtol\":0.00002,\"records\":[\n";
    size_t offset = 0;
    bool first_record = true;
    for (const case_spec & spec : gpu_cases()) {
        fixture f(spec, false);
        f.allocate(device.backend);
        need(f.accepts() == spec.eligible, "allocated host plan disagrees for " + spec.name);
        const input_values values = make_inputs(spec);
        const std::vector<double> expected = reference(spec, values);
        std::vector<float> actual(expected.size()), initial, kept_initial;
        reference_error error;
        const int calls = opts.graphs ? 12 : 2;
        for (int call = 0; call < calls; ++call) {
            upload(f, values);
            need(ggml_backend_graph_compute(device.backend, f.graph) == GGML_STATUS_SUCCESS,
                 spec.name + ": backend graph computation failed");
            ggml_backend_synchronize(device.backend);
            ggml_backend_tensor_get(f.final, actual.data(), 0, actual.size() * sizeof(float));
            error = compare_reference(expected, actual, spec.name);
            if (call == 0) { initial = actual; }
            else { compare_repeat(initial, actual, spec.name); }
            if (spec.variant == "intermediate-output" || spec.variant == "unary-output") {
                const bool keep_add = spec.variant == "intermediate-output";
                ggml_tensor * kept = keep_add ? f.add : f.unary;
                std::vector<float> intermediate(actual.size());
                ggml_backend_tensor_get(kept, intermediate.data(), 0, intermediate.size() * sizeof(float));
                (void)compare_reference(reference(spec, values, keep_add ? "add" : "unary"), intermediate,
                                        spec.name + " preserved intermediate");
                if (call == 0) { kept_initial = intermediate; }
                else { compare_repeat(kept_initial, intermediate, spec.name + " preserved intermediate"); }
            }
        }
        raw.write(reinterpret_cast<const char *>(actual.data()), std::streamsize(actual.size() * sizeof(float)));
        need(bool(raw), "raw output write failed");
        if (!first_record) { json << ",\n"; }
        first_record = false;
        json << "{\"case\":" << quoted(spec.name) << ",\"kind\":" << quoted(spec.kind)
             << ",\"cont\":" << (spec.cont ? "true" : "false")
             << ",\"pattern\":" << quoted(spec.cont ? "cont" : "direct")
             << ",\"node\":" << quoted(ggml_get_name(f.out))
             << ",\"graph_first_node\":" << quoted(ggml_get_name(f.graph->nodes[0]))
             << ",\"variant\":" << quoted(spec.variant) << ",\"width\":" << spec.width << ",\"rows\":" << spec.rows
             << ",\"data\":" << quoted(spec.data) << ",\"bias_first\":" << (spec.bias_first ? "true" : "false")
             << ",\"eligible\":" << (spec.eligible ? "true" : "false")
             << ",\"expected_candidate\":" << (spec.eligible && opts.mode ? "true" : "false")
             << ",\"raw_offset_bytes\":" << offset * sizeof(float) << ",\"offset_elements\":" << offset
             << ",\"elements\":" << actual.size() << ",\"compute_calls\":" << calls
             << ",\"finite\":true,\"nonfinite\":0,\"reference_bound_failures\":0,\"reference_max_abs\":" << error.maximum
             << ",\"reference_rmse\":" << error.rmse << ",\"repeat_bit_mismatches\":0,\"repeat_max_abs\":0}";
        offset += actual.size();
        std::cout << "ACTIVATION_CASE case=" << spec.name << " node=" << ggml_get_name(f.out)
                  << " expected_candidate=" << (spec.eligible && opts.mode ? 1 : 0)
                  << " finite=1 reference_bound_ok=1 repeat_exact=1\n";
    }
    json << "\n]}\n";
    json.close(); raw.close();
    need(bool(json) && bool(raw), "output close failed");
    std::cout << "CUDA tests passed: " << opts.prefix << ".json\n";
}
} // namespace

int main(int argc, char ** argv) {
    try {
        const options opts = parse(argc, argv);
        need(std::fegetround() == FE_TONEAREST, "F32 input ADD reference requires round-to-nearest");
        const int checks = host_checks();
        if (!opts.host_only) { gpu_tests(opts, checks); }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "bias-activation test failed: " << error.what() << '\n';
        return 1;
    }
}
