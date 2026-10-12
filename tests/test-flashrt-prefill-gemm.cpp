// SPDX-License-Identifier: MIT
// FlashRT fp16_nn prefill/vision GEMM integration fixtures.
// Outputs are checked for finiteness everywhere. Numerical reference checks
// sample 96 dispersed output coordinates using independent double dot products
// on the actual rounded operands; they are not a full-tensor reference proof.
// FP16-output native FlashRT and GGML are not required to be bitwise equal.
// The sampled numerical bound gates only mode-1 eligible native kernels.
// GGML baseline/fallback errors are measured with the same bound and recorded;
// they do not inherit the native FP32-accumulation accuracy requirement.
#include "ggml.h"
#include "ggml-cuda/flashrt-prefill-gemm-guard.h"
#ifndef FLASHRT_PREFILL_HOST_ONLY
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
static void need(bool condition, const std::string & message) {
    if (!condition) { throw std::runtime_error(message); }
}
struct options {
    bool host_only = false;
    bool graphs = false;
    int mode = -1;
    std::string prefix;
};
static options parse(int argc, char ** argv) {
    options result;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--host-only") { result.host_only = true; continue; }
        if (key == "--help") {
            std::cout << "Usage: test-flashrt-prefill-gemm --host-only\n"
                         "   or: --mode 0|1 --graphs off|on --output-prefix PATH\n";
            std::exit(0);
        }
        need(i + 1 < argc, "missing value after " + key);
        const std::string value = argv[++i];
        if (key == "--mode") {
            need(value == "0" || value == "1", "mode must be 0 or 1");
            result.mode = value == "1" ? 1 : 0;
        } else if (key == "--graphs") {
            need(value == "off" || value == "on", "graphs must be off or on");
            result.graphs = value == "on";
        } else if (key == "--output-prefix") { result.prefix = value; }
        else { throw std::runtime_error("unknown option: " + key); }
    }
    need(result.host_only || (result.mode >= 0 && !result.prefix.empty()), "mode and output-prefix are required");
    return result;
}
struct case_spec {
    std::string name;
    int k;
    int n;
    int m;
    ggml_type input_type;
    bool eligible;
    bool precision_f32;
};
static std::vector<case_spec> gpu_cases() {
    std::vector<case_spec> cases;
    auto add = [&](int k, int n, int m, ggml_type input_type, bool eligible, bool precision_f32 = false) {
        const std::string name = "pref_" + std::string(input_type == GGML_TYPE_F32 ? "f32" : "f16") +
            "_k" + std::to_string(k) + "_n" + std::to_string(n) + "_m" + std::to_string(m) +
            (precision_f32 ? "_prec32" : "");
        cases.push_back({name, k, n, m, input_type, eligible, precision_f32});
    };
    for (const auto & shape : std::vector<std::pair<int, int>>{{1152,1152}, {1152,4304}, {4304,1152}, {1152,2048}}) {
        add(shape.first, shape.second, 256, GGML_TYPE_F32, true);
    }
    for (int m : {264, 268}) {
        for (const auto & shape : std::vector<std::pair<int, int>>{{2048,2048}, {2048,256}, {2048,16384}, {16384,2048}}) {
            add(shape.first, shape.second, m, GGML_TYPE_F32, true);
        }
    }
    add(1152, 1152, 256, GGML_TYPE_F16, true);
    add(2048, 256, 2, GGML_TYPE_F32, false);
    add(2048, 256, 11, GGML_TYPE_F32, false);
    add(2048, 256, 265, GGML_TYPE_F32, false);
    add(1152, 1152, 256, GGML_TYPE_F32, false, true);
    return cases;
}
struct fixture {
    ggml_context * weights_ctx = nullptr;
    ggml_context * graph_ctx = nullptr;
    ggml_tensor * weight = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;
    ggml_cgraph * graph = nullptr;
    case_spec spec;
#ifndef FLASHRT_PREFILL_HOST_ONLY
    ggml_backend_buffer_t weights_buffer = nullptr;
    ggml_backend_buffer_t graph_buffer = nullptr;
#endif
    explicit fixture(case_spec s, bool host = true) : spec(std::move(s)) {
        weights_ctx = ggml_init({8 * ggml_tensor_overhead(), nullptr, true});
        graph_ctx = ggml_init({16 * ggml_tensor_overhead() + ggml_graph_overhead_custom(16, false), nullptr, true});
        need(weights_ctx && graph_ctx, "metadata allocation failed");
        weight = ggml_new_tensor_2d(weights_ctx, GGML_TYPE_F16, spec.k, spec.n);
        input = ggml_new_tensor_2d(graph_ctx, spec.input_type, spec.k, spec.m);
        output = ggml_mul_mat(graph_ctx, weight, input);
        if (spec.precision_f32) { ggml_mul_mat_set_prec(output, GGML_PREC_F32); }
        ggml_set_name(weight, (spec.name + "_w").c_str());
        ggml_set_name(input, (spec.name + "_x").c_str());
        ggml_set_name(output, spec.name.c_str());
        ggml_set_input(input);
        ggml_set_output(output);
        graph = ggml_new_graph_custom(graph_ctx, 16, false);
        ggml_build_forward_expand(graph, output);
        if (host) {
            // Address-only contract tests: these pointers are never accessed.
            // Separate large intervals avoid allocating 64 MiB for each guard.
            weight->data = reinterpret_cast<void *>(uintptr_t(0x10000000));
            input->data = reinterpret_cast<void *>(uintptr_t(0x40000000));
            output->data = reinterpret_cast<void *>(uintptr_t(0x70000000));
        }
    }
    fixture(const fixture &) = delete;
    fixture & operator=(const fixture &) = delete;
    ~fixture() {
#ifndef FLASHRT_PREFILL_HOST_ONLY
        if (graph_buffer) { ggml_backend_buffer_free(graph_buffer); }
        if (weights_buffer) { ggml_backend_buffer_free(weights_buffer); }
#endif
        if (graph_ctx) { ggml_free(graph_ctx); }
        if (weights_ctx) { ggml_free(weights_ctx); }
    }
    bool accepts() const {
        ggml_flashrt_prefill_plan plan{};
        return ggml_flashrt_plan_prefill_gemm(output, plan);
    }
#ifndef FLASHRT_PREFILL_HOST_ONLY
    void allocate(ggml_backend_t backend) {
        weights_buffer = ggml_backend_alloc_ctx_tensors(weights_ctx, backend);
        graph_buffer = ggml_backend_alloc_ctx_tensors(graph_ctx, backend);
        need(weights_buffer && graph_buffer, "CUDA tensor allocation failed");
        ggml_backend_buffer_set_usage(weights_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        need(ggml_nbytes(output) == size_t(spec.n) * spec.m * sizeof(float), "unexpected output extent");
    }
#endif
};
static int host_checks() {
    int count = 0;
    auto check = [&](bool actual, bool expected, const std::string & label) {
        ++count;
        need(actual == expected, "host contract: " + label);
    };
    for (const auto & spec : gpu_cases()) {
        fixture f(spec);
        check(f.accepts(), spec.eligible, spec.name);
        if (spec.eligible) {
            ggml_flashrt_prefill_plan plan{};
            const char * rejection = "unchanged";
            const bool accepted = ggml_flashrt_plan_prefill_gemm(f.output, plan, &rejection);
            check(accepted && rejection == nullptr && plan.weight == f.weight && plan.input == f.input &&
                  plan.output == f.output && plan.k == spec.k && plan.n == spec.n && plan.m == spec.m,
                  true, "plan roles, dimensions and cleared rejection reason");
        }
    }
    const case_spec base{"host", 1152, 1152, 256, GGML_TYPE_F32, true, false};
    { fixture f(base); f.output->op = GGML_OP_ADD; check(f.accepts(), false, "wrong operator"); }
    { fixture f(base); f.output->src[0] = nullptr; check(f.accepts(), false, "missing weights"); }
    { fixture f(base); f.output->src[1] = nullptr; check(f.accepts(), false, "missing activation"); }
    { fixture f(base); f.output->src[2] = f.input; check(f.accepts(), false, "unexpected third source"); }
    { fixture f(base); ggml_mul_mat_set_prec(f.output, GGML_PREC_F32); check(f.accepts(), false, "explicit F32 precision"); }
    { fixture f(base); ggml_mul_mat_set_hint(f.output, GGML_HINT_SRC0_IS_HADAMARD); check(f.accepts(), false, "special hint"); }
    { fixture f(base); f.weight->type = GGML_TYPE_F32; check(f.accepts(), false, "wrong weight dtype"); }
    { fixture f(base); f.input->type = GGML_TYPE_BF16; check(f.accepts(), false, "wrong input dtype"); }
    { fixture f(base); f.output->type = GGML_TYPE_F16; check(f.accepts(), false, "wrong output dtype"); }
    for (int which : {0, 1, 2}) {
        { fixture f(base); ggml_tensor * t = which == 0 ? f.weight : which == 1 ? f.input : f.output;
          t->data = nullptr; check(f.accepts(), false, "unallocated tensor"); }
        { fixture f(base); ggml_tensor * t = which == 0 ? f.weight : which == 1 ? f.input : f.output;
          t->data = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(t->data) + 4);
          check(f.accepts(), false, "unaligned tensor"); }
        { fixture f(base); ggml_tensor * t = which == 0 ? f.weight : which == 1 ? f.input : f.output;
          t->nb[1] += 16; check(f.accepts(), false, "row padding unsupported"); }
        { fixture f(base); ggml_tensor * t = which == 0 ? f.weight : which == 1 ? f.input : f.output;
          t->ne[2] = 2; check(f.accepts(), false, "batched matrix unsupported"); }
        { fixture f(base); ggml_tensor * t = which == 0 ? f.weight : which == 1 ? f.input : f.output;
          t->data = reinterpret_cast<void *>(std::numeric_limits<uintptr_t>::max() - 15);
          check(f.accepts(), false, "address extent overflow"); }
    }
    { fixture f(base); f.weight->view_src = f.input; check(f.accepts(), false, "weight view"); }
    { fixture f(base); f.weight->view_offs = 16; check(f.accepts(), false, "weight view offset"); }
    { fixture f(base); f.input->ne[0] -= 1; check(f.accepts(), false, "input K mismatch"); }
    { fixture f(base); f.output->ne[0] -= 1; check(f.accepts(), false, "output N mismatch"); }
    { fixture f(base); f.output->ne[1] -= 1; check(f.accepts(), false, "output M mismatch"); }
    { fixture f(base); f.weight->nb[3] += 16; check(f.accepts(), false, "invalid fourth-dimension stride"); }
    { fixture f(base); f.output->data = f.input->data; check(f.accepts(), false, "output/input exact alias"); }
    { fixture f(base); f.output->data = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(f.input->data) + 16);
      check(f.accepts(), false, "output/input partial alias"); }
    { fixture f(base); f.output->data = f.weight->data; check(f.accepts(), false, "output/weight alias"); }
    { fixture f(base); f.input->data = f.weight->data; check(f.accepts(), false, "input/weight alias"); }
    { fixture f(base); f.input->data = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(f.weight->data) + ggml_nbytes(f.weight));
      check(f.accepts(), true, "adjacent nonoverlapping tensors"); }
    { fixture f(base); f.weight->ne[0] = std::numeric_limits<int64_t>::max(); check(f.accepts(), false, "invalid huge width"); }
    { ggml_flashrt_prefill_plan plan{}; const char * rejection = nullptr;
      check(ggml_flashrt_plan_prefill_gemm(nullptr, plan, &rejection), false, "null destination");
      check(rejection != nullptr, true, "rejection reason supplied"); }
    std::cout << count << " prefill GEMM metadata/address contract checks passed; aliases are host-only\n";
    return count;
}

#ifndef FLASHRT_PREFILL_HOST_ONLY
static void set_environment(const char * key, const char * value) {
#if defined(_WIN32)
    need(_putenv_s(key, value ? value : "") == 0, key);
#else
    need((value ? setenv(key, value, 1) : unsetenv(key)) == 0, key);
#endif
}
static std::string quoted(const std::string & value) {
    std::ostringstream out;
    out << '"';
    for (const unsigned char c : value) {
        if (c == '"' || c == '\\') { out << '\\' << char(c); }
        else if (c < 32) { out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec; }
        else { out << char(c); }
    }
    out << '"';
    return out.str();
}
struct prng {
    uint32_t state;
    float next() {
        state ^= state << 13; state ^= state >> 17; state ^= state << 5;
        return float(int32_t(state & 0xfffffu) - 524288) / 524288.f;
    }
};
static std::vector<ggml_fp16_t> make_weights(const case_spec & spec) {
    std::vector<ggml_fp16_t> weights(size_t(spec.k) * spec.n);
    prng random{0x49513d57u ^ uint32_t(spec.k * 31 + spec.n * 7)};
    const float scale = 1.0f / std::sqrt(float(spec.k));
    for (auto & value : weights) { value = ggml_fp32_to_fp16(random.next() * scale); }
    return weights;
}
struct activation_values {
    std::vector<float> values;
    std::vector<ggml_fp16_t> rounded;
};
static activation_values make_activation(const case_spec & spec, bool changed) {
    activation_values x;
    x.values.resize(size_t(spec.k) * spec.m);
    x.rounded.resize(x.values.size());
    prng random{(changed ? 0x1338ab91u : 0x592ce841u) ^ uint32_t(spec.k * 13 + spec.m * 43)};
    for (size_t i = 0; i < x.values.size(); ++i) {
        x.values[i] = random.next();
        x.rounded[i] = ggml_fp32_to_fp16(x.values[i]);
        if (spec.input_type == GGML_TYPE_F16) { x.values[i] = ggml_fp16_to_fp32(x.rounded[i]); }
    }
    return x;
}
struct reference_sample { size_t index; double expected; };
static std::vector<reference_sample> sampled_reference(const case_spec & spec,
        const std::vector<ggml_fp16_t> & weights, const activation_values & x) {
    const size_t elements = size_t(spec.n) * spec.m;
    std::set<size_t> coordinates;
    for (int m : {0, 1, spec.m / 2, spec.m - 1}) {
        for (int n : {0, 1, spec.n / 2, spec.n - 1}) { coordinates.insert(size_t(m) * spec.n + n); }
    }
    uint64_t state = 0x6342185a1234ULL;
    while (coordinates.size() < std::min(size_t(96), elements)) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        coordinates.insert(size_t((state >> 13) % elements));
    }
    std::vector<reference_sample> result;
    for (size_t coordinate : coordinates) {
        const size_t m = coordinate / spec.n;
        const size_t n = coordinate % spec.n;
        double dot = 0;
        for (int k = 0; k < spec.k; ++k) {
            // Default GGML and native FlashRT use F16 activation operands.
            // The explicit-precision fallback instead checks actual F32 X.
            const double activation = spec.precision_f32 ? double(x.values[m * spec.k + k]) :
                double(ggml_fp16_to_fp32(x.rounded[m * spec.k + k]));
            dot += double(ggml_fp16_to_fp32(weights[n * spec.k + k])) * activation;
        }
        result.push_back({coordinate, dot});
    }
    return result;
}
struct reference_error { double maximum = 0; double rmse = 0; size_t bound_failures = 0; };
static reference_error check_output(const std::vector<float> & actual,
        const std::vector<reference_sample> & expected, const std::string & label, bool gate) {
    constexpr double atol = 0.004;
    constexpr double rtol = 0.002;
    need(!actual.empty() && !expected.empty(), label + ": empty output/reference");
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i])) {
            throw std::runtime_error(label + ": nonfinite output at " + std::to_string(i));
        }
    }
    reference_error error;
    double squared = 0;
    for (const auto & sample : expected) {
        need(sample.index < actual.size(), "reference coordinate outside output");
        const double difference = std::abs(double(actual[sample.index]) - sample.expected);
        const double bound = atol + rtol * std::abs(sample.expected);
        need(std::isfinite(sample.expected), "nonfinite independent reference");
        if (difference > bound) { ++error.bound_failures; }
        if (gate && difference > bound) {
            std::ostringstream message;
            message << std::setprecision(17) << label << ": sampled independent reference mismatch index="
                    << sample.index << " expected=" << sample.expected << " actual=" << actual[sample.index]
                    << " abs_error=" << difference << " bound=" << bound;
            throw std::runtime_error(message.str());
        }
        error.maximum = std::max(error.maximum, difference);
        squared += difference * difference;
    }
    error.rmse = std::sqrt(squared / double(expected.size()));
    return error;
}
static bool bit_equal(const std::vector<float> & left, const std::vector<float> & right) {
    return left.size() == right.size() && std::memcmp(left.data(), right.data(), left.size() * sizeof(float)) == 0;
}
struct backend_owner {
    ggml_backend_t backend = nullptr;
    ~backend_owner() { if (backend) { ggml_backend_synchronize(backend); ggml_backend_free(backend); } }
};
static void gpu_tests(const options & opts, int checks) {
    set_environment("GGML_CUDA_FLASHRT", "1");
    set_environment("GGML_CUDA_FLASHRT_PREFILL_GEMM", opts.mode ? "1" : "0");
    set_environment("GGML_CUDA_FLASHRT_TRACE", "1");
    set_environment("GGML_CUDA_DISABLE_GRAPHS", opts.graphs ? nullptr : "1");
    backend_owner device;
    device.backend = ggml_backend_cuda_init(0);
    need(device.backend != nullptr, "CUDA0 backend initialization failed");
    const uint16_t endian = 1;
    need(*reinterpret_cast<const unsigned char *>(&endian) == 1, "raw export requires little-endian host");
    std::ofstream raw(opts.prefix + ".bin", std::ios::binary | std::ios::trunc);
    need(bool(raw), "cannot open raw output file");
    std::ostringstream records;
    records << std::setprecision(17);
    size_t offset = 0;
    bool first_record = true;
    for (const auto & spec : gpu_cases()) {
        fixture f(spec, false);
        f.allocate(device.backend);
        need(f.accepts() == spec.eligible, "allocated plan disagrees for " + spec.name);
        const auto weights = make_weights(spec);
        ggml_backend_tensor_set(f.weight, weights.data(), 0, weights.size() * sizeof(ggml_fp16_t));
        std::vector<float> initial;
        for (int phase_index = 0; phase_index < 2; ++phase_index) {
            const bool changed = phase_index != 0;
            const std::string phase = changed ? "changed" : "initial";
            const auto x = make_activation(spec, changed);
            const auto expected = sampled_reference(spec, weights, x);
            if (spec.input_type == GGML_TYPE_F16) {
                ggml_backend_tensor_set(f.input, x.rounded.data(), 0, x.rounded.size() * sizeof(ggml_fp16_t));
            } else { ggml_backend_tensor_set(f.input, x.values.data(), 0, x.values.size() * sizeof(float)); }
            const int calls = changed ? 3 : 8;
            std::vector<float> first;
            std::vector<float> actual(size_t(spec.n) * spec.m);
            reference_error error;
            for (int call = 0; call < calls; ++call) {
                need(ggml_backend_graph_compute(device.backend, f.graph) == GGML_STATUS_SUCCESS,
                     spec.name + ": graph compute failed");
                ggml_backend_synchronize(device.backend);
                ggml_backend_tensor_get(f.output, actual.data(), 0, actual.size() * sizeof(float));
                error = check_output(actual, expected, spec.name + "/" + phase, opts.mode == 1 && spec.eligible);
                if (call == 0) { first = actual; }
                else { need(bit_equal(first, actual), spec.name + "/" + phase + ": repeated output bit mismatch"); }
            }
            if (!changed) { initial = actual; }
            else { need(!bit_equal(initial, actual), spec.name + ": input update did not change output"); }
            const size_t raw_bytes = actual.size() * sizeof(float);
            raw.write(reinterpret_cast<const char *>(actual.data()), std::streamsize(raw_bytes));
            need(bool(raw), "raw output write failed");
            if (!first_record) { records << ",\n"; }
            first_record = false;
            records << "{\"name\":" << quoted(spec.name) << ",\"node\":" << quoted(ggml_get_name(f.output))
                    << ",\"k\":" << spec.k << ",\"n\":" << spec.n << ",\"m\":" << spec.m
                    << ",\"input_type\":" << quoted(spec.input_type == GGML_TYPE_F32 ? "f32" : "f16")
                    << ",\"precision\":" << quoted(spec.precision_f32 ? "f32" : "default")
                    << ",\"eligible\":" << (spec.eligible ? "true" : "false")
                    << ",\"expected_candidate\":" << (spec.eligible && opts.mode ? "true" : "false")
                    << ",\"phase\":" << quoted(phase) << ",\"raw_dtype\":\"little-endian-f32\""
                    << ",\"raw_offset_bytes\":" << offset << ",\"raw_bytes\":" << raw_bytes
                    << ",\"elements\":" << actual.size() << ",\"compute_calls\":" << calls
                    << ",\"finite\":true,\"reference_samples\":" << expected.size()
                    << ",\"reference_gate_applied\":" << (opts.mode == 1 && spec.eligible ? "true" : "false")
                    << ",\"reference_bound_failures\":" << error.bound_failures << ",\"reference_max_abs\":" << error.maximum
                    << ",\"reference_rmse\":" << error.rmse
                    << ",\"tolerance_atol\":0.004,\"tolerance_rtol\":0.002,\"repeat_bit_mismatches\":0}";
            offset += raw_bytes;
            std::cout << "PREFILL_CASE name=" << spec.name << " phase=" << phase
                      << " expected_candidate=" << (spec.eligible && opts.mode ? 1 : 0)
                      << " finite=1 reference_samples=" << expected.size()
                      << " reference_gate_applied=" << (opts.mode == 1 && spec.eligible ? 1 : 0)
                      << " reference_bound_failures=" << error.bound_failures << " repeat_exact=1\n";
        }
    }
    raw.close();
    need(bool(raw), "raw output close failed");
    std::ofstream json(opts.prefix + ".json", std::ios::trunc);
    need(bool(json), "cannot open manifest output file");
    json << "{\"schema_version\":1,\"status\":\"tests_complete\",\"backend\":\"CUDA0\",\"mode\":" << opts.mode
         << ",\"graphs_requested\":" << (opts.graphs ? "true" : "false") << ",\"host_checks\":" << checks
         << ",\"case_count\":" << gpu_cases().size() << ",\"record_count\":" << 2 * gpu_cases().size()
         << ",\"raw_file\":" << quoted(opts.prefix + ".bin") << ",\"raw_bytes\":" << offset
         << ",\"input_update_checked\":true,\"gpu_alias_cases\":0"
         << ",\"reference\":\"96 sampled independent double dot products per phase on F16-rounded W/X; explicit-F32-precision fallback uses F32 X\""
         << ",\"reference_scope\":\"sampled_coordinates_only_all_output_elements_checked_finite\""
         << ",\"reference_gate_policy\":\"bound gates mode1 eligible native kernels only; unchanged GGML baseline/fallback errors are recorded with the same bound\""
         << ",\"comparison_policy\":\"same mode repeats and cross-graph bytes exact; modes compared numerically\""
         << ",\"records\":[\n" << records.str() << "\n]}\n";
    json.close();
    need(bool(json), "manifest output close failed");
    std::cout << "CUDA prefill GEMM tests passed: " << opts.prefix << ".json\n";
}
#endif
} // namespace
int main(int argc, char ** argv) {
    try {
        const options opts = parse(argc, argv);
        const int checks = host_checks();
#ifdef FLASHRT_PREFILL_HOST_ONLY
        (void) checks;
#endif
        if (!opts.host_only) {
#ifdef FLASHRT_PREFILL_HOST_ONLY
            throw std::runtime_error("this binary was compiled host-only; supply --host-only");
#else
            gpu_tests(opts, checks);
#endif
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "Prefill GEMM test failed: " << error.what() << '\n';
        return 1;
    }
}
