// SPDX-License-Identifier: Apache-2.0
// cuBLASLt FP16 GEMM adapter for GGML's contiguous small-M decoder matrices.
// Design reference: FlashRT GemmRunner's cached FP16 descriptors and algorithms,
// csrc/gemm/gemm_runner.cu at e881e1424cf4fadee7dff121d01cc4e159e550e7:
// https://github.com/flashrt-project/FlashRT/blob/e881e1424cf4fadee7dff121d01cc4e159e550e7/csrc/gemm/gemm_runner.cu
// FlashRT's referenced source is licensed under Apache-2.0.
// Independently implemented GGML layout/lifetime adaptation. Unlike FlashRT's
// FP16_NN output contract, this adapter writes F32 directly and does not reorder
// weights. It does not implement FlashRT's FP8 or NVFP4 paths.

#ifdef GGML_CUDA_FLASHRT
#include "flashrt-gemm.cuh"
#include "flashrt-gemm-guard.h"
#include "convert.cuh"

#include <cublasLt.h>

#include <array>
#include <cstring>
#include <memory>

namespace {

constexpr size_t gemm_workspace_bytes = 32 * 1024 * 1024;
constexpr size_t gemm_max_input_elements = 4096 * 11;

static bool gemm_env_enabled(const char * name) {
    const char * value = std::getenv(name);
    return value && std::strcmp(value, "1") == 0;
}

struct gemm_cached_plan {
    int k = 0;
    int n = 0;
    int m = 11;
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

    ~gemm_cached_plan() { destroy(); }

    gemm_cached_plan() = default;
    gemm_cached_plan(const gemm_cached_plan &) = delete;
    gemm_cached_plan & operator=(const gemm_cached_plan &) = delete;

    void initialize(cublasLtHandle_t handle, int columns_in, int columns_out) {
        k = columns_in;
        n = columns_out;
        CUBLAS_CHECK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        const cublasOperation_t trans_a = CUBLAS_OP_T;
        const cublasOperation_t trans_b = CUBLAS_OP_N;
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA,
                                                   &trans_a, sizeof(trans_a)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB,
                                                   &trans_b, sizeof(trans_b)));

        // GGML: w[k + K*n], x[k + K*m], out[n + N*m]. Default Lt order is
        // column-major, so op(A)=w^T has shape N x K and B has shape K x M.
        // No packing/reordering or hidden output downcast is needed.
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&a, CUDA_R_16F, k, n, k));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&b, CUDA_R_16F, k, m, k));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&d, CUDA_R_32F, n, m, n));

        cublasLtMatmulPreference_t preference = nullptr;
        CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
        CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference,
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &gemm_workspace_bytes, sizeof(gemm_workspace_bytes)));
        const uint32_t alignment = 16;
        const cublasLtMatmulPreferenceAttributes_t alignment_attrs[] = {
            CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
            CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
            CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
            CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES,
        };
        for (const auto attr : alignment_attrs) {
            CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference,
                attr, &alignment, sizeof(alignment)));
        }
        cublasLtMatmulHeuristicResult_t result{};
        int count = 0;
        const cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
            handle, op, a, b, d, d, preference, 1, &result, &count);
        CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
        if (status == CUBLAS_STATUS_NOT_SUPPORTED) {
            return;
        }
        CUBLAS_CHECK(status);
        if (count == 0 || result.state == CUBLAS_STATUS_NOT_SUPPORTED) {
            return;
        }
        CUBLAS_CHECK(result.state);
        if (result.workspaceSize > gemm_workspace_bytes) {
            return;
        }
        algo = result.algo;
        workspace_bytes = result.workspaceSize;
        available = true;
    }
};

static bool gemm_on_device(const ggml_tensor * tensor, int device) {
    return tensor->buffer && ggml_backend_buffer_get_type(tensor->buffer)
        == ggml_backend_cuda_buffer_type(device);
}

} // namespace

struct ggml_cuda_flashrt_gemm_state {
    int device;
    cudaStream_t stream;
    cublasLtHandle_t handle = nullptr;
    void * workspace = nullptr;
    half * input_f16 = nullptr;
    std::array<gemm_cached_plan, 5> plans;

    ggml_cuda_flashrt_gemm_state(int device_id, cudaStream_t owner_stream)
        : device(device_id), stream(owner_stream) {
        ggml_cuda_set_device(device);
        CUBLAS_CHECK(cublasLtCreate(&handle));
        CUDA_CHECK(cudaMalloc(&workspace, gemm_workspace_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&input_f16),
                              gemm_max_input_elements * sizeof(half)));
        // Allocate all bounded shape descriptors and scratch before capture.
        // We choose the cached top-1 heuristic; there is no runtime autotuning.
        plans[0].initialize(handle, 1024, 2048);
        plans[1].initialize(handle, 1024,  256);
        plans[2].initialize(handle, 2048, 1024);
        plans[3].initialize(handle, 1024, 4096);
        plans[4].initialize(handle, 4096, 1024);
    }

    ~ggml_cuda_flashrt_gemm_state() {
        const int previous_device = ggml_cuda_get_device();
        ggml_cuda_set_device(device);
        // The owning context calls reset while its capture exclusion lock is
        // still alive. A captured graph can replay on a different launch
        // stream, so synchronizing only the original stream is insufficient.
        CUDA_CHECK(cudaDeviceSynchronize());
        if (input_f16) { CUDA_CHECK(cudaFree(input_f16)); }
        if (workspace) { CUDA_CHECK(cudaFree(workspace)); }
        for (auto & plan : plans) { plan.destroy(); }
        if (handle) { CUBLAS_CHECK(cublasLtDestroy(handle)); }
        ggml_cuda_set_device(previous_device);
    }

    ggml_cuda_flashrt_gemm_state(const ggml_cuda_flashrt_gemm_state &) = delete;
    ggml_cuda_flashrt_gemm_state & operator=(const ggml_cuda_flashrt_gemm_state &) = delete;

    gemm_cached_plan * find(const ggml_flashrt_gemm_plan & tensor_plan) {
        for (auto & plan : plans) {
            if (plan.k == tensor_plan.k && plan.n == tensor_plan.n && plan.m == tensor_plan.m) {
                return &plan;
            }
        }
        return nullptr;
    }
};

bool ggml_cuda_flashrt_gemm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    static const bool enabled = gemm_env_enabled("GGML_CUDA_FLASHRT")
                             && gemm_env_enabled("GGML_CUDA_FLASHRT_GEMM");
    static const bool trace = gemm_env_enabled("GGML_CUDA_FLASHRT_TRACE");
    if (!enabled || ggml_cuda_info().devices[ctx.device].cc != GGML_CUDA_CC_ADA_LOVELACE) {
        return false;
    }
    ggml_flashrt_gemm_plan tensors;
    if (!ggml_flashrt_plan_gemm(dst, tensors) ||
        !gemm_on_device(tensors.weight, ctx.device) ||
        !gemm_on_device(tensors.input, ctx.device) ||
        !gemm_on_device(tensors.output, ctx.device) ||
        ggml_backend_buffer_get_usage(tensors.weight->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        return false;
    }
    GGML_ASSERT(ctx.curr_stream_no >= 0 && ctx.curr_stream_no < GGML_CUDA_MAX_STREAMS);
    const cudaStream_t stream = ctx.stream();
    cudaStreamCaptureStatus capture;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
    auto & state = ctx.flashrt_gemm_states[ctx.curr_stream_no];
    if (!state) {
        if (capture != cudaStreamCaptureStatusNone) {
            return false;
        }
        state = std::make_shared<ggml_cuda_flashrt_gemm_state>(ctx.device, stream);
    }
    GGML_ASSERT(state->device == ctx.device && state->stream == stream);
    gemm_cached_plan * plan = state->find(tensors);
    if (!plan || !plan->available ||
        (capture != cudaStreamCaptureStatusNone && !plan->warmed)) {
        return false;
    }
    // Up to this point all refusals leave the destination untouched. Once
    // execution starts, CUDA/cuBLAS errors must be reported, never hidden.
    const to_fp16_cuda_t convert = ggml_get_to_fp16_cuda(GGML_TYPE_F32);
    GGML_ASSERT(convert != nullptr);
    convert(tensors.input->data, state->input_f16, int64_t(tensors.k) * tensors.m, stream);
    CUDA_CHECK(cudaGetLastError());
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_CHECK(cublasLtMatmul(state->handle, plan->op,
        &alpha, tensors.weight->data, plan->a, state->input_f16, plan->b,
        &beta, tensors.output->data, plan->d, tensors.output->data, plan->d,
        &plan->algo, state->workspace, plan->workspace_bytes, stream));
    CUDA_CHECK(cudaGetLastError());
    plan->warmed = true;
    if (trace) {
        GGML_LOG_INFO("flashrt_gemm: enqueued mode=1 node=%s M=%d N=%d K=%d "
                      "weight=f16 input=f32_to_f16 accumulate=f32 output=f32 capture=%d\n",
                      dst->name, tensors.m, tensors.n, tensors.k,
                      int(capture != cudaStreamCaptureStatusNone));
    }
    return true;
}
#endif // GGML_CUDA_FLASHRT
