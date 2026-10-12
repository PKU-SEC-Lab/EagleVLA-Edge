// SPDX-License-Identifier: MIT
// Isolated CUDA backend comparison for the five pi0 decoder GEMM shapes.
// This program never promotes a candidate: the runner must verify dispatch
// traces, compare raw outputs with mode 0, and assess measured performance.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
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
#include <vector>

namespace {

struct options {
    std::string output_prefix;
    std::string case_filter = "all";
    int mode = -1;
    int warmup = 5;
    int repeat = 80;
    bool graphs = false;
    bool check_only = false;
};

static void need(bool value, const std::string & message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

static int integer(const std::string & value) {
    size_t end = 0;
    const int result = std::stoi(value, &end);
    need(end == value.size(), "invalid integer: " + value);
    return result;
}

static options parse_options(int argc, char ** argv) {
    options result;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--check-only") {
            result.check_only = true;
            continue;
        }
        if (key == "--help") {
            std::cout << "Usage: test-flashrt-gemm-bench --output-prefix PATH --mode 0|1|2 "
                         "[--warmup 5] [--repeat 80] [--case all|normal] "
                         "[--graphs off|on] [--check-only]\n";
            std::exit(0);
        }
        need(i + 1 < argc, "missing value after " + key);
        const std::string value = argv[++i];
        if (key == "--output-prefix") {
            result.output_prefix = value;
        } else if (key == "--mode") {
            result.mode = integer(value);
        } else if (key == "--warmup") {
            result.warmup = integer(value);
        } else if (key == "--repeat") {
            result.repeat = integer(value);
        } else if (key == "--case") {
            result.case_filter = value;
        } else if (key == "--graphs") {
            need(value == "on" || value == "off", "--graphs must be on or off");
            result.graphs = value == "on";
        } else {
            throw std::runtime_error("unknown option: " + key);
        }
    }
    need(!result.output_prefix.empty(), "--output-prefix is required");
    need(result.mode >= 0 && result.mode <= 2, "--mode must be 0, 1, or 2");
    need(result.warmup >= 5 && result.warmup <= 1000, "--warmup must be 5..1000");
    need(result.repeat >= 50 && result.repeat <= 10000, "--repeat must be 50..10000");
    need(result.case_filter == "all" || result.case_filter == "normal",
         "--case must be all or normal");
    return result;
}

static void environment(const char * key, const char * value) {
#if defined(_WIN32)
    need(_putenv_s(key, value ? value : "") == 0, std::string("cannot set ") + key);
#else
    need((value ? setenv(key, value, 1) : unsetenv(key)) == 0,
         std::string("cannot set ") + key);
#endif
}

static std::string json_string(const std::string & value) {
    std::ostringstream out;
    out << '"';
    for (const unsigned char ch : value) {
        if (ch == '"' || ch == '\\') {
            out << '\\' << char(ch);
        } else if (ch < 32) {
            out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(ch)
                << std::dec;
        } else {
            out << char(ch);
        }
    }
    out << '"';
    return out.str();
}

struct backend_owner {
    ggml_backend_t value = nullptr;
    ~backend_owner() {
        if (value) {
            ggml_backend_synchronize(value);
            ggml_backend_free(value);
        }
    }
};

struct graph_owner {
    ggml_context * weights_ctx = nullptr;
    ggml_context * graph_ctx = nullptr;
    ggml_backend_buffer_t weights_buffer = nullptr;
    ggml_backend_buffer_t graph_buffer = nullptr;
    ggml_tensor * weights = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;
    ggml_cgraph * graph = nullptr;

    ~graph_owner() {
        if (graph_buffer) { ggml_backend_buffer_free(graph_buffer); }
        if (weights_buffer) { ggml_backend_buffer_free(weights_buffer); }
        if (graph_ctx) { ggml_free(graph_ctx); }
        if (weights_ctx) { ggml_free(weights_ctx); }
    }

    void initialize(ggml_backend_t backend, int k, int n, const std::string & node) {
        ggml_init_params wp = {4 * ggml_tensor_overhead(), nullptr, true};
        ggml_init_params gp = {8 * ggml_tensor_overhead() + ggml_graph_overhead_custom(16, false),
                               nullptr, true};
        weights_ctx = ggml_init(wp);
        graph_ctx = ggml_init(gp);
        need(weights_ctx && graph_ctx, "cannot allocate GGML metadata contexts");
        weights = ggml_new_tensor_2d(weights_ctx, GGML_TYPE_F16, k, n);
        input = ggml_new_tensor_2d(graph_ctx, GGML_TYPE_F32, k, 11);
        output = ggml_mul_mat(graph_ctx, weights, input);
        ggml_set_name(weights, (node + "_weight").c_str());
        ggml_set_name(input, (node + "_input").c_str());
        ggml_set_name(output, node.c_str());
        ggml_set_input(input);
        ggml_set_output(output);
        graph = ggml_new_graph_custom(graph_ctx, 16, false);
        ggml_build_forward_expand(graph, output);
        // Real weight usage is part of candidate eligibility. Weights are in
        // their own device allocation; activations must not share that usage.
        weights_buffer = ggml_backend_alloc_ctx_tensors(weights_ctx, backend);
        graph_buffer = ggml_backend_alloc_ctx_tensors(graph_ctx, backend);
        need(weights_buffer && graph_buffer, "cannot allocate CUDA tensors");
        ggml_backend_buffer_set_usage(weights_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        need(ggml_nbytes(output) == size_t(n) * 11 * sizeof(float), "unexpected output layout");
    }
};

struct prng {
    uint32_t state;
    uint32_t next() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state;
    }
    float signed_value() {
        // Integer arithmetic and power-of-two scaling are deterministic;
        // std::uniform_real_distribution is not specified across libraries.
        return float(int32_t(next() & 0x000fffffu) - 524288) / 524288.0f;
    }
};

struct inputs {
    std::vector<ggml_fp16_t> weights;
    std::vector<float> weights_f32;
    std::vector<float> activation;
    std::vector<float> activation_rounded;
};

static inputs make_inputs(int k, int n, int case_index) {
    inputs data;
    data.weights.resize(size_t(k) * n);
    data.weights_f32.resize(data.weights.size());
    data.activation.resize(size_t(k) * 11);
    data.activation_rounded.resize(data.activation.size());
    prng random{0x31415927u ^ uint32_t(k * 13 + n * 29 + case_index * 997)};
    if (case_index == 1) {
        // Opposite neighbouring weight columns cancel except for a small
        // activation perturbation. This exposes reduction-order sensitivity.
        for (size_t i = 0; i < data.weights.size(); i += 2) {
            const float weight = random.signed_value() * 0.25f;
            data.weights[i] = ggml_fp32_to_fp16(weight);
            data.weights[i + 1] = ggml_fp32_to_fp16(-weight);
        }
        for (size_t i = 0; i < data.activation.size(); i += 2) {
            const float value = random.signed_value();
            data.activation[i] = value;
            data.activation[i + 1] = value + 1.0f / 4096.0f;
        }
    } else {
        for (auto & weight : data.weights) {
            weight = ggml_fp32_to_fp16(random.signed_value() * (case_index == 2 ? 4.0f : 0.25f));
        }
        for (size_t i = 0; i < data.activation.size(); ++i) {
            const float scale = case_index == 2 ? (i % 5 == 0 ? 2048.0f : 1.0f / 1024.0f) : 1.0f;
            data.activation[i] = random.signed_value() * scale;
        }
    }
    for (size_t i = 0; i < data.weights.size(); ++i) {
        data.weights_f32[i] = ggml_fp16_to_fp32(data.weights[i]);
    }
    for (size_t i = 0; i < data.activation.size(); ++i) {
        // The baseline MMF also rounds F32 activations to FP16 before MMA.
        data.activation_rounded[i] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(data.activation[i]));
    }
    return data;
}

struct result {
    std::string case_name;
    std::string node;
    int k = 0;
    int n = 0;
    size_t offset_elements = 0;
    size_t elements = 0;
    size_t nonfinite = 0;
    size_t fp64_bound_violations = 0;
    int compute_calls = 0;
    double max_abs_fp64 = 0.0;
    double rmse_fp64 = 0.0;
    double reference_scale = 0.0;
    double repeat_max_abs = 0.0;
    double p50_us = 0.0;
    double p95_us = 0.0;
    std::vector<double> measurements_us;
    bool measured = false;
};

static void compare_reference(const inputs & input, const std::vector<float> & actual, result & row) {
    double squared = 0.0;
    // A conservative FP32 dot-product forward-error bound, applied per output
    // to the actual rounded FP16 operands. This detects gross math/layout
    // errors; it does not grant mode 1 equivalence to baseline or model safety.
    const double eps = std::numeric_limits<float>::epsilon();
    const double gamma = row.k * eps / (1.0 - row.k * eps);
    for (int m = 0; m < 11; ++m) {
        for (int n = 0; n < row.n; ++n) {
            const float value = actual[size_t(m) * row.n + n];
            if (!std::isfinite(value)) {
                ++row.nonfinite;
                continue;
            }
            double reference = 0.0;
            double sum_abs = 0.0;
            for (int k = 0; k < row.k; ++k) {
                const double product = double(input.weights_f32[size_t(n) * row.k + k]) *
                                       double(input.activation_rounded[size_t(m) * row.k + k]);
                reference += product;
                sum_abs += std::abs(product);
            }
            const double error = std::abs(double(value) - reference);
            row.max_abs_fp64 = std::max(row.max_abs_fp64, error);
            squared += error * error;
            row.reference_scale = std::max(row.reference_scale, sum_abs);
            if (error > 4.0 * gamma * sum_abs + 1.0e-7) {
                ++row.fp64_bound_violations;
            }
        }
    }
    row.rmse_fp64 = std::sqrt(squared / double(actual.size()));
}

static double quantile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    const double index = fraction * double(values.size() - 1);
    const size_t lo = size_t(index);
    const size_t hi = std::min(lo + 1, values.size() - 1);
    return values[lo] + (values[hi] - values[lo]) * (index - double(lo));
}

static void compute(ggml_backend_t backend, graph_owner & graph, result & row) {
    const ggml_status status = ggml_backend_graph_compute_async(backend, graph.graph);
    ggml_backend_synchronize(backend);
    need(status == GGML_STATUS_SUCCESS, "backend graph compute failed for " + row.node);
    ++row.compute_calls;
}

static void compare_repeat(const std::vector<float> & first, const std::vector<float> & later,
                           result & row) {
    for (size_t i = 0; i < first.size(); ++i) {
        if (!std::isfinite(first[i]) || !std::isfinite(later[i])) {
            ++row.nonfinite;
        } else {
            row.repeat_max_abs = std::max(row.repeat_max_abs, std::abs(double(first[i]) - later[i]));
        }
    }
}

static result run_case(const options & opts, ggml_backend_t backend, int k, int n,
                       int case_index, size_t offset, std::ofstream & binary) {
    const char * names[] = {"normal", "cancellation", "bounded"};
    result row;
    row.case_name = names[case_index];
    row.k = k;
    row.n = n;
    row.elements = size_t(n) * 11;
    row.offset_elements = offset;
    row.node = "gemm_K" + std::to_string(k) + "_N" + std::to_string(n) + "_" + row.case_name;
    std::cerr << "flashrt_gemm_bench: begin case=" << row.case_name
              << " M=11 N=" << n << " K=" << k << std::endl;
    graph_owner graph;
    graph.initialize(backend, k, n, row.node);
    const inputs host = make_inputs(k, n, case_index);
    ggml_backend_tensor_set(graph.weights, host.weights.data(), 0, host.weights.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(graph.input, host.activation.data(), 0, host.activation.size() * sizeof(float));
    ggml_backend_synchronize(backend);
    std::vector<float> first(row.elements);
    std::vector<float> output(row.elements);
    compute(backend, graph, row);
    ggml_backend_tensor_get(graph.output, first.data(), 0, first.size() * sizeof(float));
    output = first;
    // CUDA graph warmup is backend/version dependent. Twelve calls offer a
    // capture/replay exercise; actual graph evidence must come from logs or
    // profiling, not from an assumed correspondence between calls and replay.
    const int checks = opts.graphs ? 12 : 1;
    for (int i = 1; i < checks; ++i) {
        compute(backend, graph, row);
        ggml_backend_tensor_get(graph.output, output.data(), 0, output.size() * sizeof(float));
        compare_repeat(first, output, row);
    }
    if (!opts.check_only && case_index == 0) {
        for (int i = 0; i < opts.warmup; ++i) {
            compute(backend, graph, row);
        }
        std::vector<double> elapsed;
        elapsed.reserve(size_t(opts.repeat));
        for (int i = 0; i < opts.repeat; ++i) {
            const auto begin = std::chrono::steady_clock::now();
            compute(backend, graph, row);
            const auto end = std::chrono::steady_clock::now();
            elapsed.push_back(std::chrono::duration<double, std::micro>(end - begin).count());
        }
        ggml_backend_tensor_get(graph.output, output.data(), 0, output.size() * sizeof(float));
        compare_repeat(first, output, row);
        row.p50_us = quantile(elapsed, 0.50);
        row.p95_us = quantile(elapsed, 0.95);
        row.measurements_us = elapsed;
        row.measured = true;
    }
    compare_reference(host, output, row);
    // Only the final execution is exported, while repeat_max_abs above proves
    // that capture and repeated execution retained the first output's values.
    binary.write(reinterpret_cast<const char *>(output.data()), std::streamsize(output.size() * sizeof(float)));
    need(bool(binary), "cannot write output tensor file");
    std::cerr << row.node << ": finite=" << (row.nonfinite == 0)
              << " fp64_bound_pass=" << (row.fp64_bound_violations == 0 && row.nonfinite == 0)
              << " repeat_max_abs=" << row.repeat_max_abs;
    if (row.measured) {
        std::cerr << " backend_graph_sync_p50_us=" << row.p50_us;
    }
    std::cerr << '\n';
    std::cerr << "flashrt_gemm_bench: end case=" << row.case_name
              << " M=11 N=" << n << " K=" << k << std::endl;
    return row;
}

static void write_result(const options & opts, const std::string & backend,
                         const std::vector<result> & records, bool passed) {
    std::ofstream out(opts.output_prefix + ".json", std::ios::binary);
    need(bool(out), "cannot create output JSON");
    out << std::setprecision(17) << std::boolalpha;
    out << "{\n  \"schema_version\": 1,\n  \"mode\": " << opts.mode
        << ",\n  \"backend\": " << json_string(backend)
        << ",\n  \"graphs_requested\": " << opts.graphs
        << ",\n  \"warmup\": " << opts.warmup << ",\n  \"repeat\": " << opts.repeat
        << ",\n  \"check_only\": " << opts.check_only
        << ",\n  \"case_filter\": " << json_string(opts.case_filter)
        << ",\n  \"binary_dtype\": \"f32le\",\n  \"input_generator_version\": 1,"
        << "\n  \"timing_metric\": \"synchronized_backend_graph_wall_us\","
        << "\n  \"candidate_dispatch_verified\": false,"
        << "\n  \"numeric_sanity_pass\": " << passed << ",\n  \"records\": [\n";
    for (size_t i = 0; i < records.size(); ++i) {
        const result & row = records[i];
        if (i != 0) { out << ",\n"; }
        out << "    {\"case\": " << json_string(row.case_name)
            << ", \"node\": " << json_string(row.node)
            << ", \"k\": " << row.k << ", \"n\": " << row.n << ", \"m\": 11"
            << ", \"offset_elements\": " << row.offset_elements
            << ", \"elements\": " << row.elements
            << ", \"compute_calls\": " << row.compute_calls
            << ", \"nonfinite\": " << row.nonfinite
            << ", \"max_abs_fp64\": " << row.max_abs_fp64
            << ", \"rmse_fp64\": " << row.rmse_fp64
            << ", \"reference_scale\": " << row.reference_scale
            << ", \"fp64_bound_violations\": " << row.fp64_bound_violations
            << ", \"fp64_bound_pass\": " << (row.nonfinite == 0 && row.fp64_bound_violations == 0)
            << ", \"repeat_max_abs\": " << row.repeat_max_abs
            << ", \"p50_us\": ";
        if (row.measured) { out << row.p50_us; } else { out << "null"; }
        out << ", \"p95_us\": ";
        if (row.measured) { out << row.p95_us; } else { out << "null"; }
        out << ", \"measurements_us\": [";
        for (size_t sample = 0; sample < row.measurements_us.size(); ++sample) {
            if (sample != 0) { out << ", "; }
            out << row.measurements_us[sample];
        }
        out << "]";
        out << "}";
    }
    out << "\n  ]\n}\n";
    out.flush();
    need(bool(out), "cannot write output JSON");
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const options opts = parse_options(argc, argv);
        const uint32_t endian = 1;
        need(*reinterpret_cast<const unsigned char *>(&endian) == 1 && sizeof(float) == 4 &&
             std::numeric_limits<float>::is_iec559, "raw tensor protocol requires little-endian IEEE F32");
        need(!std::ifstream(opts.output_prefix + ".json").good() &&
             !std::ifstream(opts.output_prefix + ".bin").good(), "output files already exist; use a new prefix");
        if (!opts.check_only) {
            for (const char * key : {"GGML_CUDA_FLASHRT_TRACE", "GGML_CUDA_FLASHRT_GEMM_GRAPH_TRACE"}) {
                const char * value = std::getenv(key);
                need(!value || std::strcmp(value, "1") != 0,
                     std::string("timing requires ") + key + "=0; use --check-only for dispatch tracing");
            }
        }
        environment("GGML_CUDA_FLASHRT", "1");
        environment("GGML_CUDA_FLASHRT_GEMM", std::to_string(opts.mode).c_str());
        environment("GGML_CUDA_DISABLE_GRAPHS", opts.graphs ? nullptr : "1");
        environment("GGML_CUDA_FLASHRT_SHADOW", "0");
        environment("GGML_CUDA_FLASHRT_GEMM_SHADOW", "0");
        need(ggml_backend_cuda_get_device_count() > 0, "no CUDA device is visible");
        backend_owner backend;
        backend.value = ggml_backend_cuda_init(0);
        need(backend.value != nullptr, "cannot initialize CUDA0 backend");
        std::ofstream binary(opts.output_prefix + ".bin", std::ios::binary);
        need(bool(binary), "cannot create output tensor file; create parent directory first");
        const int shapes[][2] = {{1024, 2048}, {1024, 256}, {2048, 1024}, {1024, 4096}, {4096, 1024}};
        std::vector<result> records;
        size_t offset = 0;
        bool passed = true;
        for (const auto & shape : shapes) {
            for (int c = 0; c < (opts.case_filter == "normal" ? 1 : 3); ++c) {
                const result row = run_case(opts, backend.value, shape[0], shape[1], c, offset, binary);
                offset += row.elements;
                passed = passed && row.nonfinite == 0 && row.fp64_bound_violations == 0 && row.repeat_max_abs == 0.0;
                records.push_back(row);
            }
        }
        binary.close();
        need(bool(binary), "cannot finalize output tensor file");
        write_result(opts, ggml_backend_name(backend.value), records, passed);
        std::cout << "saved: " << opts.output_prefix << ".json and .bin\n";
        std::cout << "numeric_sanity_pass=" << (passed ? "true" : "false")
                  << "; candidate dispatch and mode-0 equivalence require the runner\n";
        return passed ? 0 : 2;
    } catch (const std::exception & error) {
        std::cerr << "GEMM bench failed: " << error.what() << '\n';
        return 1;
    }
}
