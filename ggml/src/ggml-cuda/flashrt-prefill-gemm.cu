// SPDX-License-Identifier: Apache-2.0
// Adapted from FlashRT GemmRunner::fp16_nn and its FP16_NN descriptor cache,
// csrc/gemm/gemm_runner.cu at e881e1424cf4fadee7dff121d01cc4e159e550e7:
// https://github.com/flashrt-project/FlashRT/blob/e881e1424cf4fadee7dff121d01cc4e159e550e7/csrc/gemm/gemm_runner.cu
// Complete upstream license: vendor/flashrt/prefill-gemm/LICENSE.
//
// Changes: GGML column-major transpose adaptation instead of repacking weights;
// explicit F32-to-F16 activation and F16-to-GGML-F32 output boundary kernels;
// SM89 and bounded prefill/vision shape dispatch; per-context/per-stream owned
// scratch and descriptor lifetime; pre-capture preparation with no autotuning;
// a 32 MiB workspace cap in place of upstream's 256 MiB. The native FP16 operand,
// FP32 compute and FP16 core output contract is preserved, including the final
// FP16 rounding before extending to GGML F32. Different layouts/heuristics and
// accumulation may change numerical results versus either runtime. This is
// not a bitwise GGML mode, an FP8 path, or a new handwritten GEMM core.

#ifdef GGML_CUDA_FLASHRT

#include "flashrt-prefill-gemm.cuh"
#include "flashrt-prefill-gemm-guard.h"
#include "common.cuh"

#include <cublasLt.h>

#include <array>
#include <cstdlib>
#include <cstring>
#include <memory>

namespace {

constexpr size_t prefill_workspace_bytes = 32 * 1024 * 1024;
constexpr size_t prefill_max_elements = 268 * 16384;

static bool prefill_env_one(const char * name) {
    const char * value = std::getenv(name);
    return value && std::strcmp(value, "1") == 0;
}

static bool prefill_on_device(const ggml_tensor * tensor, int device) {
    if (!tensor || !tensor->buffer || ggml_backend_buffer_get_type(tensor->buffer)
        != ggml_backend_cuda_buffer_type(device)) {
        return false;
    }
    // The metadata guard has proved nb[3] is the contiguous tensor extent.
    // Also constrain it to this allocation before reading any CUDA memory.
    const uintptr_t base = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(tensor->buffer));
    const uintptr_t begin = reinterpret_cast<uintptr_t>(tensor->data);
    const size_t size = ggml_backend_buffer_get_size(tensor->buffer);
    return base != 0 && begin >= base && begin - base <= size &&
        tensor->nb[3] <= size - size_t(begin - base);
}

static bool prefill_reject(ggml_tensor * dst, const char * reason, bool trace) {
    if (trace && dst) {
        GGML_LOG_INFO("flashrt_prefill_gemm_skip: node=%s reason=%s\n", dst->name, reason);
    }
    return false;
}

static __global__ void flashrt_prefill_input_to_f16(const float * input, half * output, int elements) {
    const int i = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (i < elements) { output[i] = __float2half_rn(input[i]); }
}

static __global__ void flashrt_prefill_output_to_f32(const half * input, float * output, int elements) {
    const int i = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (i < elements) { output[i] = __half2float(input[i]); }
}

struct prefill_cached_plan {
    int k = 0;
    int n = 0;
    int m = 0;
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a = nullptr;
    cublasLtMatrixLayout_t b = nullptr;
    cublasLtMatrixLayout_t d = nullptr;
    cublasLtMatmulAlgo_t algo{};
    size_t workspace_bytes = 0;
    bool available = false;
    bool warmed = false;

    void destroy() {
        if (d) { CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(d)); d = nullptr; }
        if (b) { CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(b)); b = nullptr; }
        if (a) { CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(a)); a = nullptr; }
        if (op) { CUBLAS_CHECK(cublasLtMatmulDescDestroy(op)); op = nullptr; }
    }

    ~prefill_cached_plan() { destroy(); }
    prefill_cached_plan() = default;
    prefill_cached_plan(const prefill_cached_plan &) = delete;
    prefill_cached_plan & operator=(const prefill_cached_plan &) = delete;

    void initialize(cublasLtHandle_t handle, int columns_in, int columns_out, int rows) {
        k = columns_in;
        n = columns_out;
        m = rows;
        // FlashRT FP16_NN uses FP16 A/B/D, FP32 accumulation and host F32 scales.
        CUBLAS_CHECK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        const cublasOperation_t transpose_a = CUBLAS_OP_T;
        const cublasOperation_t transpose_b = CUBLAS_OP_N;
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA,
                                                  &transpose_a, sizeof(transpose_a)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB,
                                                  &transpose_b, sizeof(transpose_b)));
        // GGML W[n*K+k], X[m*K+k], D[m*N+n]. In Lt column-major storage,
        // A is K x N, op(A) is N x K, B is K x M and D is N x M.
        // Thus D[n + N*m] = sum_k W[k + K*n]*X[k + K*m], with no W reorder.
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&a, CUDA_R_16F, k, n, k));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&b, CUDA_R_16F, k, m, k));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&d, CUDA_R_16F, n, m, n));

        cublasLtMatmulPreference_t preference = nullptr;
        CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
        CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference,
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &prefill_workspace_bytes, sizeof(prefill_workspace_bytes)));
        const uint32_t alignment = 16;
        const cublasLtMatmulPreferenceAttributes_t attrs[] = {
            CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
            CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
            CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
            CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES,
        };
        for (const auto attr : attrs) {
            CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference, attr,
                                                            &alignment, sizeof(alignment)));
        }
        cublasLtMatmulHeuristicResult_t result{};
        int count = 0;
        const cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
            handle, op, a, b, d, d, preference, 1, &result, &count);
        CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
        if (status == CUBLAS_STATUS_NOT_SUPPORTED) { return; }
        CUBLAS_CHECK(status);
        if (count == 0 || result.state == CUBLAS_STATUS_NOT_SUPPORTED) { return; }
        CUBLAS_CHECK(result.state);
        if (result.workspaceSize > prefill_workspace_bytes) { return; }
        algo = result.algo;
        workspace_bytes = result.workspaceSize;
        available = true;
    }
};

} // namespace

struct ggml_cuda_flashrt_prefill_state {
    int device;
    cudaStream_t stream;
    cublasLtHandle_t handle = nullptr;
    void * workspace = nullptr;
    half * input_f16 = nullptr;
    half * output_f16 = nullptr;
    std::array<prefill_cached_plan, 12> plans;

    ggml_cuda_flashrt_prefill_state(int device_id, cudaStream_t owner_stream)
        : device(device_id), stream(owner_stream) {
        ggml_cuda_set_device(device);
        CUBLAS_CHECK(cublasLtCreate(&handle));
        CUDA_CHECK(cudaMalloc(&workspace, prefill_workspace_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&input_f16), prefill_max_elements * sizeof(half)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&output_f16), prefill_max_elements * sizeof(half)));
        // Prepare every bounded shape outside capture. No handle creation,
        // allocation, heuristic query or descriptor mutation occurs in capture.
        plans[0].initialize(handle, 1152, 1152, 256);
        plans[1].initialize(handle, 1152, 4304, 256);
        plans[2].initialize(handle, 4304, 1152, 256);
        plans[3].initialize(handle, 1152, 2048, 256);
        size_t index = 4;
        for (const int rows : {264, 268}) {
            plans[index++].initialize(handle, 2048, 2048, rows);
            plans[index++].initialize(handle, 2048, 256, rows);
            plans[index++].initialize(handle, 2048, 16384, rows);
            plans[index++].initialize(handle, 16384, 2048, rows);
        }
    }

    ~ggml_cuda_flashrt_prefill_state() {
        const int previous_device = ggml_cuda_get_device();
        ggml_cuda_set_device(device);
        // The context explicitly resets this shared_ptr while holding its
        // capture exclusion lock. Captured graphs may replay on another stream;
        // synchronizing the originally recorded stream alone is insufficient.
        CUDA_CHECK(cudaDeviceSynchronize());
        if (output_f16) { CUDA_CHECK(cudaFree(output_f16)); }
        if (input_f16) { CUDA_CHECK(cudaFree(input_f16)); }
        if (workspace) { CUDA_CHECK(cudaFree(workspace)); }
        for (auto & plan : plans) { plan.destroy(); }
        if (handle) { CUBLAS_CHECK(cublasLtDestroy(handle)); }
        ggml_cuda_set_device(previous_device);
    }

    ggml_cuda_flashrt_prefill_state(const ggml_cuda_flashrt_prefill_state &) = delete;
    ggml_cuda_flashrt_prefill_state & operator=(const ggml_cuda_flashrt_prefill_state &) = delete;

    prefill_cached_plan * find(const ggml_flashrt_prefill_plan & tensor_plan) {
        for (auto & plan : plans) {
            if (plan.k == tensor_plan.k && plan.n == tensor_plan.n && plan.m == tensor_plan.m) {
                return &plan;
            }
        }
        return nullptr;
    }
};

bool ggml_cuda_flashrt_prefill_gemm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    static const bool enabled = prefill_env_one("GGML_CUDA_FLASHRT") &&
        prefill_env_one("GGML_CUDA_FLASHRT_PREFILL_GEMM");
    static const bool trace = prefill_env_one("GGML_CUDA_FLASHRT_TRACE");
    if (!enabled) { return false; }
    if (ggml_cuda_info().devices[ctx.device].cc != GGML_CUDA_CC_ADA_LOVELACE) {
        return prefill_reject(dst, "device_not_sm89", trace);
    }
    // An explicit user override is a precision contract; never silently ignore
    // it or reinterpret it as permission to use the FlashRT FP32 compute mode.
    if (std::getenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE") != nullptr) {
        return prefill_reject(dst, "explicit_cublas_compute_override", trace);
    }
    ggml_flashrt_prefill_plan tensors{};
    const char * reason = nullptr;
    if (!ggml_flashrt_plan_prefill_gemm(dst, tensors, &reason)) {
        return prefill_reject(dst, reason, trace);
    }
    if (!prefill_on_device(tensors.weight, ctx.device) ||
        !prefill_on_device(tensors.input, ctx.device) || !prefill_on_device(tensors.output, ctx.device)) {
        return prefill_reject(dst, "device_buffer", trace);
    }
    if (ggml_backend_buffer_get_usage(tensors.weight->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        return prefill_reject(dst, "weight_buffer_usage", trace);
    }
    GGML_ASSERT(ctx.curr_stream_no >= 0 && ctx.curr_stream_no < GGML_CUDA_MAX_STREAMS);
    const cudaStream_t stream = ctx.stream();
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
    auto & state = ctx.flashrt_prefill_gemm_states[ctx.curr_stream_no];
    if (!state) {
        if (capture != cudaStreamCaptureStatusNone) {
            return prefill_reject(dst, "state_not_prepared_before_capture", trace);
        }
        state = std::make_shared<ggml_cuda_flashrt_prefill_state>(ctx.device, stream);
    }
    GGML_ASSERT(state->device == ctx.device && state->stream == stream);
    prefill_cached_plan * plan = state->find(tensors);
    if (!plan || !plan->available) {
        return prefill_reject(dst, "no_supported_lt_heuristic", trace);
    }
    if (capture != cudaStreamCaptureStatusNone && !plan->warmed) {
        return prefill_reject(dst, "shape_not_warmed_before_capture", trace);
    }
    const int input_elements = tensors.k * tensors.m;
    const int output_elements = tensors.n * tensors.m;
    GGML_ASSERT(size_t(input_elements) <= prefill_max_elements &&
                size_t(output_elements) <= prefill_max_elements);
    // Refusals above have not written dst. After the first enqueue, errors are
    // reported through CUDA/CUBLAS_CHECK, never hidden by a fallback return.
    const half * input = static_cast<const half *>(tensors.input->data);
    if (tensors.input->type == GGML_TYPE_F32) {
        flashrt_prefill_input_to_f16<<<(input_elements + 255) / 256, 256, 0, stream>>>(
            static_cast<const float *>(tensors.input->data), state->input_f16, input_elements);
        CUDA_CHECK(cudaGetLastError());
        input = state->input_f16;
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_CHECK(cublasLtMatmul(state->handle, plan->op,
        &alpha, tensors.weight->data, plan->a, input, plan->b,
        &beta, state->output_f16, plan->d, state->output_f16, plan->d,
        &plan->algo, state->workspace, plan->workspace_bytes, stream));
    CUDA_CHECK(cudaGetLastError());
    flashrt_prefill_output_to_f32<<<(output_elements + 255) / 256, 256, 0, stream>>>(
        state->output_f16, static_cast<float *>(tensors.output->data), output_elements);
    CUDA_CHECK(cudaGetLastError());
    plan->warmed = true;
    if (trace) {
        GGML_LOG_INFO("flashrt_prefill_gemm: M=%d N=%d K=%d input=%s core_output=f16 "
                      "accumulate=f32 output=f32 node=%s weight=%s capture=%d domain=%s\n",
                      tensors.m, tensors.n, tensors.k,
                      tensors.input->type == GGML_TYPE_F16 ? "f16" : "f32",
                      dst->name, tensors.weight->name,
                      int(capture != cudaStreamCaptureStatusNone),
                      tensors.m == 256 ? "vision" : "encoder");
    }
    return true;
}

#endif // GGML_CUDA_FLASHRT
