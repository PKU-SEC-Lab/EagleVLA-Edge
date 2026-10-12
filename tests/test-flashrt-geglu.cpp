// SPDX-License-Identifier: MIT
// FlashRT GEGLU numerical/dispatch fixtures. The reference is deliberately
// independent of either CUDA formula: double tanh or stable double sigmoid.
// Mode 0 versus mode 1 is NOT required to be bitwise identical. Repeated
// computations, including replay after input updates, must be bitwise stable.
#include "ggml.h"
#include "ggml-cuda/flashrt-geglu-guard.h"
#ifndef FLASHRT_GEGLU_HOST_ONLY
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
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
static void need(bool ok, const std::string & message) {
    if (!ok) { throw std::runtime_error(message); }
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
            std::cout << "Usage: test-flashrt-geglu --host-only\n"
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
    ggml_type type;
    int width;
    int rows;
    std::string layout;
    bool eligible;
};
static std::vector<case_spec> gpu_cases() {
    std::vector<case_spec> cases;
    for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
        const std::string dtype = type == GGML_TYPE_F32 ? "f32" : "f16";
        cases.push_back({dtype + "_split_ae", type, 4096, 11, "split", true});
        cases.push_back({dtype + "_split_encoder", type, 16384, 257, "split", true});
        cases.push_back({dtype + "_merged", type, 4096, 11, "merged", true});
        cases.push_back({dtype + "_swapped", type, 4096, 11, "swapped", true});
        cases.push_back({dtype + "_padded_rows", type, 4096, 11, "padded_rows", true});
        cases.push_back({dtype + "_fallback_width", type, 1024, 11, "split", false});
    }
    return cases;
}
struct fixture {
    ggml_context * ctx = nullptr;
    ggml_tensor * gate_storage = nullptr;
    ggml_tensor * up_storage = nullptr;
    ggml_tensor * gate = nullptr;
    ggml_tensor * up = nullptr;
    ggml_tensor * out = nullptr;
    ggml_cgraph * graph = nullptr;
    case_spec spec;
#ifndef FLASHRT_GEGLU_HOST_ONLY
    ggml_backend_buffer_t buffer = nullptr;
#endif
    explicit fixture(case_spec s, bool host = true) : spec(std::move(s)) {
        ctx = ggml_init({host ? size_t(64 * 1024 * 1024) : size_t(256 * 1024), nullptr, !host});
        need(ctx != nullptr, "GGML metadata allocation failed");
        const bool merged = spec.layout == "merged" || spec.layout == "swapped";
        const int gate_padding = spec.layout == "padded_rows" ? 8 : 0;
        const int up_padding = spec.layout == "padded_rows" ? 16 : 0;
        gate_storage = ggml_new_tensor_2d(ctx, spec.type, merged ? 2 * spec.width : spec.width + gate_padding, spec.rows);
        gate = gate_padding ? ggml_view_2d(ctx, gate_storage, spec.width, spec.rows, gate_storage->nb[1], 0) : gate_storage;
        if (merged) {
            out = spec.layout == "swapped" ? ggml_geglu_swapped(ctx, gate) : ggml_geglu(ctx, gate);
        } else {
            up_storage = ggml_new_tensor_2d(ctx, spec.type, spec.width + up_padding, spec.rows);
            up = up_padding ? ggml_view_2d(ctx, up_storage, spec.width, spec.rows, up_storage->nb[1], 0) : up_storage;
            out = ggml_geglu_split(ctx, gate, up);
        }
        ggml_set_name(gate_storage, ("geglu_test_" + spec.name + "_gate_storage").c_str());
        if (gate != gate_storage) { ggml_set_name(gate, ("geglu_test_" + spec.name + "_gate").c_str()); }
        if (up_storage) { ggml_set_name(up_storage, ("geglu_test_" + spec.name + "_up_storage").c_str()); }
        if (up && up != up_storage) { ggml_set_name(up, ("geglu_test_" + spec.name + "_up").c_str()); }
        ggml_set_name(out, ("geglu_test_" + spec.name + "_out").c_str());
        ggml_set_input(gate_storage);
        if (up_storage) { ggml_set_input(up_storage); }
        ggml_set_output(out);
        graph = ggml_new_graph_custom(ctx, 32, false);
        ggml_build_forward_expand(graph, out);
    }
    fixture(const fixture &) = delete;
    fixture & operator=(const fixture &) = delete;
    ~fixture() {
#ifndef FLASHRT_GEGLU_HOST_ONLY
        if (buffer) { ggml_backend_buffer_free(buffer); }
#endif
        if (ctx) { ggml_free(ctx); }
    }
    bool accepts() const {
        ggml_flashrt_geglu_plan plan{};
        return ggml_flashrt_plan_geglu(out, plan);
    }
#ifndef FLASHRT_GEGLU_HOST_ONLY
    void allocate(ggml_backend_t backend) {
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        need(buffer != nullptr, "CUDA tensor allocation failed");
    }
#endif
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
            ggml_flashrt_geglu_plan p{};
            const char * reason = "unchanged";
            check(ggml_flashrt_plan_geglu(f.out, p, &reason) && reason == nullptr, true, "accept clears rejection reason");
            const bool merged = spec.layout == "merged" || spec.layout == "swapped";
            const bool swapped = spec.layout == "swapped";
            const size_t element_size = spec.type == GGML_TYPE_F32 ? 4 : 2;
            check(p.gate == f.gate && p.up == (merged ? f.gate : f.up) && p.output == f.out &&
                  p.width == spec.width && p.rows == spec.rows && p.elements == spec.width * spec.rows &&
                  p.merged == merged && p.swapped == swapped &&
                  p.gate_row_stride == f.gate->nb[1] / element_size &&
                  p.up_row_stride == (merged ? f.gate : f.up)->nb[1] / element_size &&
                  p.gate_offset == (swapped ? size_t(spec.width) : 0) &&
                  p.up_offset == (merged && !swapped ? size_t(spec.width) : 0), true, "plan preserves roles/strides/offsets");
        }
    }
    const case_spec s{"host", GGML_TYPE_F32, 4096, 11, "split", true};
    { fixture f(s); f.out->op = GGML_OP_UNARY; check(f.accepts(), false, "wrong operator"); }
    { fixture f(s); f.out->op_params[0] = GGML_GLU_OP_GEGLU_ERF; check(f.accepts(), false, "ERF is distinct"); }
    { fixture f(s); f.out->op_params[0] = GGML_GLU_OP_GEGLU_QUICK; check(f.accepts(), false, "QUICK is distinct"); }
    { fixture f(s); f.out->src[2] = f.gate; check(f.accepts(), false, "unexpected extra input"); }
    { fixture f(s); f.out->src[0] = nullptr; check(f.accepts(), false, "missing gate"); }
    { fixture f(s); f.out->data = f.gate->data; check(f.accepts(), false, "output aliases gate"); }
    { fixture f(s); f.out->data = f.up->data; check(f.accepts(), false, "output aliases up"); }
    { fixture f(s); f.out->data = static_cast<char *>(f.gate->data) + 4; check(f.accepts(), false, "partial output/gate overlap"); }
    { fixture f(s); f.out->data = nullptr; check(f.accepts(), false, "unallocated output"); }
    { fixture f(s); f.up->type = GGML_TYPE_F16; check(f.accepts(), false, "mixed input types"); }
    { fixture f(s); f.out->type = GGML_TYPE_BF16; check(f.accepts(), false, "unsupported output type"); }
    { fixture f(s); f.gate->nb[0] = 8; check(f.accepts(), false, "strided columns"); }
    { fixture f(s); f.out->nb[1] += 4; check(f.accepts(), false, "padded output"); }
    { fixture f(s); f.up->ne[1] += 1; check(f.accepts(), false, "unequal input rows"); }
    { fixture f(s); f.gate->ne[2] = 2; check(f.accepts(), false, "unsupported third dimension"); }
    { fixture f(s); f.out->op_params[1] = 2; check(f.accepts(), false, "invalid swapped encoding"); }
    { fixture f(s); f.gate->nb[1] = std::numeric_limits<size_t>::max() - 3; check(f.accepts(), false, "row extent overflow"); }
    { fixture f(s); f.out->ne[1] = std::numeric_limits<int>::max(); check(f.accepts(), false, "element count overflow"); }
    { fixture f(s); f.up->data = f.gate->data; check(f.accepts(), true, "input/input read-only alias"); }
    { fixture f(s); f.out->op_params[1] = 1; check(f.accepts(), true, "split swapped bit preserves GGML semantics"); }
    { ggml_flashrt_geglu_plan p{}; const char * reason = nullptr;
      check(ggml_flashrt_plan_geglu(nullptr, p, &reason), false, "null destination");
      check(reason != nullptr, true, "rejection reason supplied"); }
    std::cout << checks << " GEGLU metadata/address contract checks passed; alias cases are host-only\n";
    return checks;
}

#ifndef FLASHRT_GEGLU_HOST_ONLY
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
    for (unsigned char ch : value) {
        if (ch == '"' || ch == '\\') { out << '\\' << char(ch); }
        else if (ch < 32) { out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(ch) << std::dec; }
        else { out << char(ch); }
    }
    out << '"';
    return out.str();
}
struct input_values { std::vector<float> gate, up; };
static float rounded_input(ggml_type type, float value) {
    return type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(ggml_fp32_to_fp16(value)) : value;
}
static input_values make_inputs(const case_spec & spec, bool changed) {
    input_values v;
    const size_t n = size_t(spec.width) * spec.rows;
    v.gate.resize(n);
    v.up.resize(n);
    uint32_t state = changed ? 0x2f739ae1u : 0x64153791u;
    auto random = [&]() {
        state ^= state << 13; state ^= state >> 17; state ^= state << 5;
        return float(int(state & 0xfffffu) - 524288) / 524288.0f;
    };
    const float tails[] = {-100.f, -40.f, -12.f, -8.f, -6.f, -3.f, -1.f, -0.f,
                           0.f, 1e-7f, 1.f, 3.f, 6.f, 8.f, 12.f, 40.f, 100.f};
    for (size_t i = 0; i < n; ++i) {
        float g = 6.f * random();
        float u = 3.f * random();
        if (i % 53 == 0) {
            const size_t j = i / 53 + (changed ? 5 : 0);
            g = tails[j % (sizeof(tails) / sizeof(tails[0]))];
            u = (j % 2 == 0) ? -1.75f : 2.5f;
        }
        if (i % 97 == 0) { g = (i % 194 == 0) ? -0.f : 0.f; }
        if (i % 101 == 0) { u = (i % 202 == 0) ? -0.f : 0.f; }
        v.gate[i] = rounded_input(spec.type, g);
        v.up[i] = rounded_input(spec.type, u);
    }
    return v;
}
static double gelu_reference(double gate) {
    const double z = std::sqrt(2.0 / std::acos(-1.0)) * gate * (1.0 + 0.044715 * gate * gate);
    // Double tanh gives a compact independent oracle in the central region.
    // Stable double sigmoid keeps negative-tail information after tanh would
    // round exactly to -1. No CUDA float constants or expf are reused here.
    if (std::abs(z) <= 8.0) { return 0.5 * gate * (1.0 + std::tanh(z)); }
    const double x = 2.0 * z;
    const double sig = x >= 0 ? 1.0 / (1.0 + std::exp(-x)) : std::exp(x) / (1.0 + std::exp(x));
    return gate * sig;
}
static std::vector<double> reference(const input_values & v) {
    std::vector<double> result(v.gate.size());
    for (size_t i = 0; i < result.size(); ++i) { result[i] = gelu_reference(v.gate[i]) * double(v.up[i]); }
    return result;
}
static std::vector<unsigned char> encoded(const std::vector<float> & input, ggml_type type) {
    const size_t element_size = type == GGML_TYPE_F32 ? 4 : 2;
    std::vector<unsigned char> raw(input.size() * element_size);
    if (type == GGML_TYPE_F32) { std::memcpy(raw.data(), input.data(), raw.size()); }
    else {
        for (size_t i = 0; i < input.size(); ++i) {
            const ggml_fp16_t value = ggml_fp32_to_fp16(input[i]);
            std::memcpy(raw.data() + i * sizeof(value), &value, sizeof(value));
        }
    }
    return raw;
}
static void upload(fixture & f, const input_values & v) {
    const size_t width = size_t(f.spec.width);
    const size_t gate_pitch = size_t(f.gate_storage->ne[0]);
    std::vector<float> gate_data(gate_pitch * f.spec.rows, 123.0f);
    const bool merged = f.up == nullptr;
    const bool swapped = f.spec.layout == "swapped";
    for (int row = 0; row < f.spec.rows; ++row) {
        float * dst = gate_data.data() + size_t(row) * gate_pitch;
        std::copy_n(v.gate.data() + size_t(row) * width, width, dst + (swapped ? width : 0));
        if (merged) { std::copy_n(v.up.data() + size_t(row) * width, width, dst + (swapped ? 0 : width)); }
    }
    const auto gate_bytes = encoded(gate_data, f.spec.type);
    ggml_backend_tensor_set(f.gate_storage, gate_bytes.data(), 0, gate_bytes.size());
    if (!merged) {
        const size_t up_pitch = size_t(f.up_storage->ne[0]);
        std::vector<float> up_data(up_pitch * f.spec.rows, -321.0f);
        for (int row = 0; row < f.spec.rows; ++row) {
            std::copy_n(v.up.data() + size_t(row) * width, width, up_data.data() + size_t(row) * up_pitch);
        }
        const auto up_bytes = encoded(up_data, f.spec.type);
        ggml_backend_tensor_set(f.up_storage, up_bytes.data(), 0, up_bytes.size());
    }
}
struct reference_error { double maximum = 0; double rmse = 0; };
static reference_error compare_reference(const std::vector<double> & expected,
        const std::vector<unsigned char> & actual, ggml_type type, const std::string & label) {
    const size_t element_size = type == GGML_TYPE_F32 ? 4 : 2;
    need(actual.size() == expected.size() * element_size && !expected.empty(), "reference size mismatch");
    const double tolerance = type == GGML_TYPE_F32 ? 2e-5 : 2e-3;
    reference_error error;
    double squares = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        float value;
        if (type == GGML_TYPE_F32) { std::memcpy(&value, actual.data() + i * element_size, element_size); }
        else {
            ggml_fp16_t half;
            std::memcpy(&half, actual.data() + i * element_size, element_size);
            value = ggml_fp16_to_fp32(half);
        }
        need(std::isfinite(value) && std::isfinite(expected[i]), label + ": nonfinite output at " + std::to_string(i));
        const double difference = std::abs(double(value) - expected[i]);
        const double bound = tolerance * (1.0 + std::abs(expected[i]));
        if (difference > bound) {
            std::ostringstream message;
            message << label << ": independent reference mismatch at " << i << std::setprecision(17)
                    << " expected=" << expected[i] << " actual=" << value << " abs_error=" << difference << " bound=" << bound;
            throw std::runtime_error(message.str());
        }
        error.maximum = std::max(error.maximum, difference);
        squares += difference * difference;
    }
    error.rmse = std::sqrt(squares / double(expected.size()));
    return error;
}
struct backend_owner {
    ggml_backend_t backend = nullptr;
    ~backend_owner() { if (backend) { ggml_backend_synchronize(backend); ggml_backend_free(backend); } }
};
static void gpu_tests(const options & opts, int checks) {
    set_environment("GGML_CUDA_FLASHRT", "1");
    set_environment("GGML_CUDA_FLASHRT_GEGLU", opts.mode ? "1" : "0");
    set_environment("GGML_CUDA_FLASHRT_TRACE", "1");
    set_environment("GGML_CUDA_FLASHRT_SHADOW", "0");
    set_environment("GGML_CUDA_DISABLE_FUSION", "0");
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
    for (const case_spec & spec : gpu_cases()) {
        fixture f(spec, false);
        f.allocate(device.backend);
        need(f.accepts() == spec.eligible, "allocated host plan disagrees for " + spec.name);
        std::vector<unsigned char> original;
        for (int phase_index = 0; phase_index < 2; ++phase_index) {
            const bool changed = phase_index != 0;
            const std::string phase = changed ? "changed" : "initial";
            const input_values inputs = make_inputs(spec, changed);
            const auto expected = reference(inputs);
            upload(f, inputs);
            const int calls = changed ? 3 : 8;
            std::vector<unsigned char> first, actual(ggml_nbytes(f.out));
            reference_error error;
            for (int call = 0; call < calls; ++call) {
                need(ggml_backend_graph_compute(device.backend, f.graph) == GGML_STATUS_SUCCESS,
                     spec.name + ": graph compute failed");
                ggml_backend_synchronize(device.backend);
                ggml_backend_tensor_get(f.out, actual.data(), 0, actual.size());
                error = compare_reference(expected, actual, spec.type, spec.name + "/" + phase);
                if (call == 0) { first = actual; }
                else { need(first == actual, spec.name + "/" + phase + ": repeated output bit mismatch"); }
            }
            if (!changed) { original = actual; }
            else { need(original != actual, spec.name + ": input update did not change the output"); }
            raw.write(reinterpret_cast<const char *>(actual.data()), std::streamsize(actual.size()));
            need(bool(raw), "raw output write failed");
            if (!first_record) { records << ",\n"; }
            first_record = false;
            const char * dtype = spec.type == GGML_TYPE_F32 ? "f32" : "f16";
            const double tolerance = spec.type == GGML_TYPE_F32 ? 2e-5 : 2e-3;
            records << "{\"case\":" << quoted(spec.name) << ",\"name\":" << quoted(spec.name)
                    << ",\"phase\":" << quoted(phase) << ",\"dtype\":" << quoted(dtype)
                    << ",\"raw_dtype\":" << quoted(std::string("little-endian-") + dtype)
                    << ",\"layout\":" << quoted(spec.layout) << ",\"node\":" << quoted(ggml_get_name(f.out))
                    << ",\"width\":" << spec.width << ",\"rows\":" << spec.rows
                    << ",\"eligible\":" << (spec.eligible ? "true" : "false")
                    << ",\"expected_candidate\":" << (spec.eligible && opts.mode ? "true" : "false")
                    << ",\"raw_offset_bytes\":" << offset << ",\"raw_bytes\":" << actual.size()
                    << ",\"elements\":" << expected.size() << ",\"compute_calls\":" << calls
                    << ",\"finite\":true,\"reference_bound_failures\":0,\"reference_max_abs\":" << error.maximum
                    << ",\"reference_rmse\":" << error.rmse << ",\"tolerance_atol\":" << tolerance
                    << ",\"tolerance_rtol\":" << tolerance << ",\"repeat_bit_mismatches\":0}";
            offset += actual.size();
            std::cout << "GEGLU_CASE case=" << spec.name << " phase=" << phase << " dtype=" << dtype
                      << " node=" << ggml_get_name(f.out) << " expected_candidate=" << (spec.eligible && opts.mode ? 1 : 0)
                      << " finite=1 reference_bound_ok=1 repeat_exact=1\n";
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
         << ",\"gpu_alias_cases\":0,\"input_update_checked\":true"
         << ",\"reference\":\"independent double tanh / stable sigmoid GEGLU on dtype-rounded inputs\""
         << ",\"comparison_policy\":\"same mode repeats and cross-graph bytes exact; modes compared numerically\""
         << ",\"records\":[\n" << records.str() << "\n]}\n";
    json.close();
    need(bool(json), "manifest output close failed");
    std::cout << "CUDA GEGLU tests passed: " << opts.prefix << ".json\n";
}
#endif
} // namespace

int main(int argc, char ** argv) {
    try {
        const options opts = parse(argc, argv);
        const int checks = host_checks();
#ifdef FLASHRT_GEGLU_HOST_ONLY
        (void)checks;
#endif
        if (!opts.host_only) {
#ifdef FLASHRT_GEGLU_HOST_ONLY
            throw std::runtime_error("this binary was compiled host-only; supply --host-only");
#else
            gpu_tests(opts, checks);
#endif
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "GEGLU test failed: " << error.what() << '\n';
        return 1;
    }
}
