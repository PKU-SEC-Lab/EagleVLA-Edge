// SPDX-License-Identifier: MIT
// CUDA graph tests and host-only recognition tests for the SM89 F32
// (x + bias) + residual adapter. Alias mutations below are host contract
// checks; they are deliberately not reported as GPU alias executions.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-cuda/flashrt-bias-residual-guard.h"

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
            std::cout << "Usage: test-flashrt-bias-residual --host-only\n"
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
    int rows = 50;
    bool cont = false;
    bool bias_first = false;
    bool residual_first = false;
    std::string variant = "normal";
    std::string data = "random";
    bool eligible = true;
};

struct fixture {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * x = nullptr;
    ggml_tensor * x_storage = nullptr;
    ggml_tensor * bias = nullptr;
    ggml_tensor * residual = nullptr;
    ggml_tensor * add = nullptr;
    ggml_tensor * cont = nullptr;
    ggml_tensor * out = nullptr;
    ggml_tensor * final = nullptr;
    ggml_cgraph * graph = nullptr;
    int index = -1;
    case_spec spec;

    explicit fixture(case_spec s, bool host = true) : spec(std::move(s)) {
        ctx = ggml_init({host ? size_t(24 * 1024 * 1024) : size_t(256 * 1024), nullptr, !host});
        need(ctx != nullptr, "GGML metadata allocation failed");
        const bool strided = spec.variant == "strided-x";
        x_storage = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, strided ? 2304 : 1152, spec.rows);
        x = strided ? ggml_view_2d(ctx, x_storage, 1152, spec.rows, x_storage->nb[1], 0) : x_storage;
        bias = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1152,
                                 spec.variant == "wrong-bias-shape" ? spec.rows : 1);
        residual = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1152, spec.rows);
        add = spec.bias_first ? ggml_add(ctx, bias, x) : ggml_add(ctx, x, bias);
        ggml_tensor * added = add;
        if (spec.cont) {
            cont = ggml_cont(ctx, spec.variant == "wrong-cont-edge" ? x : add);
            added = cont;
        }
        if (spec.variant == "wrong-edge") { added = x; }
        out = spec.residual_first ? ggml_add(ctx, residual, added) : ggml_add(ctx, added, residual);
        final = spec.variant == "extra-consumer" ? ggml_add(ctx, out, add) : out;
        ggml_set_name(x, ("bias_test_" + spec.name + "_x").c_str());
        ggml_set_name(bias, (spec.name + "_bias").c_str());
        ggml_set_name(residual, (spec.name + "_residual").c_str());
        ggml_set_name(add, ("bias_test_" + spec.name + "_add").c_str());
        if (cont) { ggml_set_name(cont, (spec.name + "_cont").c_str()); }
        ggml_set_name(out, ("bias_test_" + spec.name + "_out").c_str());
        if (final != out) { ggml_set_name(final, (spec.name + "_final").c_str()); }
        ggml_set_input(x_storage);
        ggml_set_input(bias);
        ggml_set_input(residual);
        ggml_set_output(final);
        if (spec.variant == "intermediate-output") { ggml_set_output(add); }
        if (spec.variant == "cont-output" && cont) { ggml_set_output(cont); }
        graph = ggml_new_graph_custom(ctx, 32, false);
        if (spec.variant == "wrong-edge" || spec.variant == "wrong-cont-edge") {
            ggml_build_forward_expand(graph, add);
        }
        ggml_build_forward_expand(graph, final);
        ggml_set_name(graph->nodes[0], ("bias_test_" + spec.name + "_first").c_str());
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
        ggml_flashrt_bias_residual_plan plan{};
        return ggml_flashrt_plan_bias_residual(graph, index, plan) == (spec.cont ? 2 : 1);
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
    for (bool cont : {false, true}) {
        for (int rows : {1, 50, 256, 512}) {
            fixture f({"valid", rows, cont});
            check(f.accepts(), true, "valid dimensions");
        }
        for (const std::string variant : {"extra-consumer", "intermediate-output", "wrong-edge",
                                          "strided-x", "wrong-bias-shape"}) {
            fixture f({variant, 50, cont, false, false, variant});
            check(f.accepts(), false, variant);
        }
        for (bool first : {false, true}) {
            fixture f({"order", 1, cont, true, first});
            check(f.accepts(), true, "bias/residual operand order");
        }
        { fixture f({"out-x", 50, cont}); f.out->data = f.x->data;
          check(f.accepts(), true, "output exactly aliases x"); }
        { fixture f({"out-r", 50, cont}); f.out->data = f.residual->data;
          check(f.accepts(), true, "output exactly aliases residual"); }
        { fixture f({"a-x", 50, cont}); f.add->data = f.x->data; f.x->flags &= ~GGML_TENSOR_FLAG_INPUT;
          check(f.accepts(), true, "add exactly aliases x"); }
        { fixture f({"a-input-x", 50, cont}); f.add->data = f.x->data;
          check(f.accepts(), false, "omitted ADD write to observable INPUT x"); }
        { fixture f({"a-input-x-out", 50, cont}); f.add->data = f.x->data; f.out->data = f.x->data;
          check(f.accepts(), true, "output writes final value to aliased INPUT x"); }
        { fixture f({"a-x-later-read", 50, cont}); f.add->data = f.x->data; f.x->flags &= ~GGML_TENSOR_FLAG_INPUT;
          ggml_build_forward_expand(f.graph, ggml_add(f.ctx, f.final, f.x));
          check(f.accepts(), false, "later x consumer observes omitted ADD write"); }
        { fixture f({"out-x-part", 50, cont}); f.out->data = static_cast<char *>(f.x->data) + 8;
          check(f.accepts(), false, "output partly overlaps x"); }
        { fixture f({"out-r-part", 50, cont}); f.out->data = static_cast<char *>(f.residual->data) + 8;
          check(f.accepts(), false, "output partly overlaps residual"); }
        { fixture f({"out-bias", 50, cont}); f.out->data = f.bias->data;
          check(f.accepts(), false, "output overlaps bias"); }
        { fixture f({"a-r", 50, cont}); f.add->data = f.residual->data;
          check(f.accepts(), false, "add aliases residual before residual read"); }
        { fixture f({"a-bias", 50, cont}); f.add->data = f.bias->data;
          check(f.accepts(), false, "add aliases shared bias"); }
        { fixture f({"unaligned-bias", 50, cont}); f.bias->data = static_cast<char *>(f.bias->data) + 4;
          check(f.accepts(), false, "bias float2 misalignment"); }
        { fixture f({"unaligned-x", 50, cont}); f.x->data = static_cast<char *>(f.x->data) + 4;
          check(f.accepts(), false, "x float2 misalignment"); }
        { fixture f({"unaligned-r", 50, cont}); f.residual->data = static_cast<char *>(f.residual->data) + 4;
          check(f.accepts(), false, "residual float2 misalignment"); }
        { fixture f({"missing-data", 50, cont}); f.out->data = nullptr;
          check(f.accepts(), false, "unallocated output"); }
        { fixture f({"wrong-type", 50, cont}); f.bias->type = GGML_TYPE_F16;
          check(f.accepts(), false, "bias dtype"); }
        { fixture f({"not-computed", 50, cont}); f.add->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
          check(f.accepts(), false, "initial ADD not computed"); }
        { fixture f({"index", 50, cont}); ggml_flashrt_bias_residual_plan plan{};
          check(ggml_flashrt_plan_bias_residual(f.graph, -1, plan) != 0, false, "negative index");
          check(ggml_flashrt_plan_bias_residual(f.graph, f.graph->n_nodes, plan) != 0, false, "past end"); }
    }
    for (const std::string variant : {"wrong-cont-edge", "cont-output"}) {
        fixture f({variant, 50, true, false, false, variant});
        check(f.accepts(), false, variant);
    }
    { fixture f({"strided-cont", 50, true}); f.cont->nb[1] += 8;
      check(f.accepts(), false, "noncontiguous CONT output"); }
    { fixture f({"c-r", 50, true}); f.cont->data = f.residual->data;
      check(f.accepts(), false, "CONT overwrites residual"); }
    { fixture f({"c-bias", 50, true}); f.cont->data = f.bias->data;
      check(f.accepts(), false, "CONT overwrites bias"); }
    { fixture f({"a-c", 50, true}); f.cont->data = f.add->data;
      check(f.accepts(), true, "CONT exactly aliases ADD"); }
    { fixture f({"raw-view", 50, false});
      ggml_tensor * view = ggml_view_tensor(f.ctx, f.add);
      ggml_build_forward_expand(f.graph, ggml_add(f.ctx, f.out, view));
      check(f.accepts(), false, "raw ADD view without src edge"); }
    { fixture f({"raw-cont-view", 50, true});
      ggml_tensor * view = ggml_view_tensor(f.ctx, f.cont);
      ggml_build_forward_expand(f.graph, ggml_add(f.ctx, f.out, view));
      check(f.accepts(), false, "raw CONT view without src edge"); }
    { fixture f({"bad-rows", 2, false}); check(f.accepts(), false, "unsupported row count"); }
    std::cout << checks << " bias/residual contract checks passed; alias cases are host-only\n";
    return checks;
}

// Force a float rounding point after each addition. Do not replace this with
// a double expression or reassociate the two operations.
static float add_f32(float a, float b) { volatile float sum = a + b; return sum; }
static uint32_t bits(float value) { uint32_t out; std::memcpy(&out, &value, sizeof(out)); return out; }
struct input_values { std::vector<float> x, bias, residual; };
static input_values make_inputs(const case_spec & spec) {
    input_values values;
    const size_t count = size_t(1152) * spec.rows;
    values.x.resize(count); values.residual.resize(count);
    values.bias.resize(spec.variant == "wrong-bias-shape" ? count : 1152);
    uint32_t state = 0x19560731u;
    auto random = [&]() {
        state ^= state << 13; state ^= state >> 17; state ^= state << 5;
        return float(int(state & 0xfffffu) - 524288) / 262144.0f;
    };
    for (size_t i = 0; i < values.bias.size(); ++i) {
        values.bias[i] = spec.data == "cancellation" ? ((i & 1) ? -1.0e8f : 1.0e8f)
                       : spec.data == "signed-zero" ? ((i & 1) ? -0.0f : 0.0f) : random();
    }
    for (size_t i = 0; i < count; ++i) {
        if (spec.data == "cancellation") {
            values.x[i] = -values.bias[i % values.bias.size()];
            values.residual[i] = float(int(i % 17) - 8) / 4.0f;
        } else if (spec.data == "signed-zero") {
            values.x[i] = (i & 2) ? -0.0f : 0.0f;
            values.residual[i] = (i & 4) ? -0.0f : 0.0f;
        } else {
            values.x[i] = random() * ((i % 5) ? 0.0625f : 128.0f);
            values.residual[i] = random() * ((i % 7) ? 0.00390625f : 4.0f);
        }
    }
    return values;
}
static std::vector<float> reference(const case_spec & spec, const input_values & values, bool intermediate) {
    std::vector<float> out(values.x.size());
    for (size_t i = 0; i < out.size(); ++i) {
        const float b = values.bias[i % values.bias.size()];
        const float a = spec.bias_first ? add_f32(b, values.x[i]) : add_f32(values.x[i], b);
        if (intermediate) { out[i] = a; continue; }
        const float middle = spec.variant == "wrong-edge" || spec.variant == "wrong-cont-edge" ? values.x[i] : a;
        out[i] = spec.residual_first ? add_f32(values.residual[i], middle) : add_f32(middle, values.residual[i]);
        if (spec.variant == "extra-consumer") { out[i] = add_f32(out[i], a); }
    }
    return out;
}
static void upload(fixture & f, const input_values & values) {
    if (f.spec.variant == "strided-x") {
        std::vector<float> padded(size_t(2304) * f.spec.rows, 123.0f);
        for (int row = 0; row < f.spec.rows; ++row) {
            std::memcpy(padded.data() + size_t(row) * 2304, values.x.data() + size_t(row) * 1152, 1152 * sizeof(float));
        }
        ggml_backend_tensor_set(f.x_storage, padded.data(), 0, padded.size() * sizeof(float));
    } else { ggml_backend_tensor_set(f.x, values.x.data(), 0, values.x.size() * sizeof(float)); }
    ggml_backend_tensor_set(f.bias, values.bias.data(), 0, values.bias.size() * sizeof(float));
    ggml_backend_tensor_set(f.residual, values.residual.data(), 0, values.residual.size() * sizeof(float));
}
static std::vector<case_spec> gpu_cases() {
    std::vector<case_spec> cases;
    for (bool cont : {false, true}) {
        const std::string pattern = cont ? "cont" : "direct";
        for (int rows : {1, 50, 256, 512}) {
            cases.push_back({pattern + "_r" + std::to_string(rows), rows, cont});
        }
        for (const std::string data : {"cancellation", "signed-zero"}) {
            cases.push_back({pattern + "_" + data, 50, cont, false, false, "normal", data});
        }
        cases.push_back({pattern + "_bias-first", 1, cont, true});
        cases.push_back({pattern + "_residual-first", 50, cont, false, true});
        cases.push_back({pattern + "_both-first", 1, cont, true, true});
        for (const std::string variant : {"extra-consumer", "intermediate-output", "wrong-edge",
                                          "strided-x", "wrong-bias-shape"}) {
            cases.push_back({pattern + "_fallback_" + variant, 50, cont, false, false, variant, "random", false});
        }
    }
    for (const std::string variant : {"wrong-cont-edge", "cont-output"}) {
        cases.push_back({"cont_fallback_" + variant, 50, true, false, false, variant, "random", false});
    }
    return cases;
}
struct backend_owner {
    ggml_backend_t backend = nullptr;
    ~backend_owner() { if (backend) { ggml_backend_synchronize(backend); ggml_backend_free(backend); } }
};
static void compare(const std::vector<float> & expected, const std::vector<float> & actual,
                    const std::string & label) {
    need(expected.size() == actual.size(), "reference shape mismatch");
    for (size_t i = 0; i < actual.size(); ++i) {
        need(std::isfinite(actual[i]), label + ": nonfinite output at " + std::to_string(i));
        if (bits(expected[i]) != bits(actual[i])) {
            std::ostringstream message;
            message << label << ": sequential F32 reference mismatch at " << i
                    << " expected=" << std::setprecision(9) << expected[i] << " actual=" << actual[i]
                    << " expected_bits=" << std::hex << bits(expected[i]) << " actual_bits=" << bits(actual[i]);
            throw std::runtime_error(message.str());
        }
    }
}
static void gpu_tests(const options & opts, int checks) {
    env("GGML_CUDA_FLASHRT", "1");
    env("GGML_CUDA_FLASHRT_BIAS_RESIDUAL", opts.mode ? "1" : "0");
    env("GGML_CUDA_FLASHRT_TRACE", "1");
    env("GGML_CUDA_FLASHRT_BIAS_GRAPH_TRACE", "1");
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
    json << "{\"schema_version\":1,\"status\":\"tests_complete\",\"backend\":\"CUDA0\",\"mode\":" << opts.mode
         << ",\"graphs_requested\":" << (opts.graphs ? "true" : "false")
         << ",\"host_checks\":" << checks
         << ",\"case_count\":" << gpu_cases().size()
         << ",\"raw_dtype\":\"little-endian-f32\",\"raw_file\":" << quoted(opts.prefix + ".bin")
         << ",\"gpu_alias_cases\":0,\"reference\":\"two sequential round-to-nearest F32 additions; bitwise including signed zero\",\"records\":[\n";
    size_t offset = 0;
    bool first_record = true;
    for (const case_spec & spec : gpu_cases()) {
        fixture f(spec, false);
        f.allocate(device.backend);
        need(f.accepts() == spec.eligible, "allocated host plan disagrees for " + spec.name);
        const input_values values = make_inputs(spec);
        const std::vector<float> expected = reference(spec, values, false);
        std::vector<float> actual(expected.size()), initial;
        const int calls = opts.graphs ? 12 : 2;
        for (int call = 0; call < calls; ++call) {
            upload(f, values);
            need(ggml_backend_graph_compute(device.backend, f.graph) == GGML_STATUS_SUCCESS,
                 spec.name + ": backend graph computation failed");
            ggml_backend_synchronize(device.backend);
            ggml_backend_tensor_get(f.final, actual.data(), 0, actual.size() * sizeof(float));
            compare(expected, actual, spec.name);
            if (call == 0) { initial = actual; }
            else { compare(initial, actual, spec.name + " repeat"); }
            if (spec.variant == "intermediate-output" || spec.variant == "cont-output") {
                ggml_tensor * kept = spec.variant == "intermediate-output" ? f.add : f.cont;
                std::vector<float> intermediate(actual.size());
                ggml_backend_tensor_get(kept, intermediate.data(), 0, intermediate.size() * sizeof(float));
                compare(reference(spec, values, true), intermediate, spec.name + " preserved intermediate");
            }
        }
        raw.write(reinterpret_cast<const char *>(actual.data()), std::streamsize(actual.size() * sizeof(float)));
        need(bool(raw), "raw output write failed");
        if (!first_record) { json << ",\n"; }
        first_record = false;
        json << "{\"case\":" << quoted(spec.name) << ",\"pattern\":" << quoted(spec.cont ? "cont" : "direct")
             << ",\"node\":" << quoted(ggml_get_name(f.out))
             << ",\"graph_first_node\":" << quoted(ggml_get_name(f.graph->nodes[0]))
             << ",\"variant\":" << quoted(spec.variant) << ",\"width\":1152,\"rows\":" << spec.rows
             << ",\"eligible\":" << (spec.eligible ? "true" : "false")
             << ",\"expected_candidate\":" << (spec.eligible && opts.mode ? "true" : "false")
             << ",\"raw_offset_bytes\":" << offset * sizeof(float) << ",\"offset_elements\":" << offset
             << ",\"elements\":" << actual.size() << ",\"compute_calls\":" << calls
             << ",\"finite\":true,\"nonfinite\":0,\"reference_bit_mismatches\":0,\"repeat_bit_mismatches\":0,\"repeat_max_abs\":0}";
        offset += actual.size();
        std::cout << "BIAS_CASE case=" << spec.name << " node=" << ggml_get_name(f.out)
                  << " expected_candidate=" << (spec.eligible && opts.mode ? 1 : 0) << " exact=1\n";
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
        need(std::fegetround() == FE_TONEAREST, "CPU F32 reference requires round-to-nearest");
        const int checks = host_checks();
        if (!opts.host_only) { gpu_tests(opts, checks); }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "bias-residual test failed: " << error.what() << '\n';
        return 1;
    }
}
