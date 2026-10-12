// Paired-process benchmark for the complete NORM -> MUL -> ADD graph.
// Uses identical deterministic inputs in baseline and candidate processes.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static void check(bool ok, const char * message) {
    if (!ok) {
        std::fprintf(stderr, "%s\n", message);
        std::exit(1);
    }
}

int main(int argc, char ** argv) {
    const char * backend_name = argc > 1 ? argv[1] : "CUDA0";
    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_name(backend_name, nullptr);
    check(backend != nullptr, "Requested backend unavailable (no automatic CPU fallback)");
    std::printf("backend,rows,width,max_abs_error,p50_us,p95_us\n");
    for (int rows : { 1, 50, 256, 512 }) {
        constexpr int width = 1152;
        constexpr float eps = 1e-6f;
        ggml_init_params params = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(params);
        check(ctx != nullptr, "ggml_init failed");
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, rows);
        ggml_tensor * w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, width);
        ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, width);
        ggml_tensor * out = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, eps), w), b);
        ggml_set_name(out, "flashrt_bench_out");
        ggml_set_output(out);
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, out);
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        check(buffer != nullptr, "Tensor allocation failed");
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
        std::vector<float> input(rows * width), weight(width), bias(width), actual(rows * width);
        for (float & value : input) { value = distribution(rng); }
        for (float & value : weight) { value = distribution(rng); }
        for (float & value : bias) { value = distribution(rng); }
        ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
        ggml_backend_tensor_set(w, weight.data(), 0, weight.size() * sizeof(float));
        ggml_backend_tensor_set(b, bias.data(), 0, bias.size() * sizeof(float));
        auto compute = [&] {
            check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "Graph compute failed");
            ggml_backend_synchronize(backend);
        };
        for (int i = 0; i < 100; ++i) { compute(); }
        ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
        double max_abs = 0;
        for (int row = 0; row < rows; ++row) {
            double mean = 0, variance = 0;
            for (int col = 0; col < width; ++col) { mean += input[row * width + col]; }
            mean /= width;
            for (int col = 0; col < width; ++col) {
                const double delta = input[row * width + col] - mean;
                variance += delta * delta;
            }
            const double inverse = 1.0 / std::sqrt(variance / width + eps);
            for (int col = 0; col < width; ++col) {
                const double expected = (input[row * width + col] - mean) * inverse * weight[col] + bias[col];
                const double error = std::abs(actual[row * width + col] - expected);
                check(std::isfinite(actual[row * width + col]) && error <= 5e-5 + 5e-5 * std::abs(expected),
                      "LayerNorm differs from independent double-precision reference");
                max_abs = std::max(max_abs, error);
            }
        }
        std::vector<double> times;
        for (int i = 0; i < 200; ++i) {
            const auto start = std::chrono::steady_clock::now();
            compute();
            times.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count());
        }
        std::sort(times.begin(), times.end());
        std::printf("%s,%d,%d,%.9g,%.3f,%.3f\n", backend_name, rows, width, max_abs, times[100], times[190]);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
    ggml_backend_free(backend);
}
