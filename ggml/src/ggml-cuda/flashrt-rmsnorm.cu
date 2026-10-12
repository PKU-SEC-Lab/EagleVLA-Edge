// SPDX-License-Identifier: MIT
// FP32 host adaptation of FlashRT Pi0 RTX's RMSNorm/affine fusion region.
// Arithmetic/reduction follow ggml-cuda/norm.cu; this is not FlashRT's native
// FP16 packed-pair reduction or FP8 weight folding.
#ifdef GGML_CUDA_FLASHRT
#include "flashrt-rmsnorm.cuh"
#include "flashrt-rmsnorm-guard.h"
#include "binbcast.cuh"
#include "norm.cuh"

#include <algorithm>
#include <cstdlib>
#include <vector>

namespace {

static bool enabled_env(const char * name) {
    const char * value = std::getenv(name);
    return value && std::strcmp(value, "1") == 0;
}

template <int block_size>
static __global__ void flashrt_gemma_rmsnorm_ggml_math_f32(
        const float * x, const float * weight, float * dst, int ncols, float eps) {
    ggml_cuda_pdl_lc();
    x += int64_t(blockIdx.x) * ncols;
    dst += int64_t(blockIdx.x) * ncols;
    const int tid = threadIdx.x;
    float sum = 0.0f;
    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        sum += xi * xi;
    }
    extern __shared__ float shared[];
    sum = block_reduce<block_reduce_method::SUM, block_size>(sum, shared);
    const float mean = sum / ncols;
    const float scale = rsqrtf(mean + eps);
    for (int col = tid; col < ncols; col += block_size) {
        const float normalized = __fmul_rn(scale, x[col]);
        // Keep separate FP32 rounding: n*(1+w) or fmaf(n,w,n) would change it.
        dst[col] = __fadd_rn(__fmul_rn(normalized, weight[col]), normalized);
    }
}

static void launch(const ggml_flashrt_rmsnorm_plan & p, float * out, cudaStream_t stream) {
    const float * x = static_cast<const float *>(p.x->data);
    const float * w = static_cast<const float *>(p.weight->data);
    // Match unfused RMSNorm dispatch, including the <1024 case (256 threads).
    if (p.dim < 1024) {
        flashrt_gemma_rmsnorm_ggml_math_f32<256><<<int(p.rows), 256, 32 * sizeof(float), stream>>>(
            x, w, out, p.dim, p.eps);
    } else {
        flashrt_gemma_rmsnorm_ggml_math_f32<1024><<<int(p.rows), 1024, 32 * sizeof(float), stream>>>(
            x, w, out, p.dim, p.eps);
    }
    CUDA_CHECK(cudaGetLastError());
}

static void shadow(ggml_backend_cuda_context & ctx, const ggml_flashrt_rmsnorm_plan & p) {
    if (std::getenv("GGML_CUDA_DISABLE_GRAPHS") == nullptr) {
        GGML_ABORT("FlashRT RMSNorm shadow requires GGML_CUDA_DISABLE_GRAPHS=1");
    }
    const cudaStream_t stream = ctx.stream();
    cudaStreamCaptureStatus capture;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
    GGML_ASSERT(capture == cudaStreamCaptureStatusNone);
    const size_t n = size_t(p.rows) * size_t(p.dim);
    ggml_cuda_pool_alloc<float> scratch(ctx.pool(), 4 * n);
    ggml_tensor ref_norm = *p.norm;
    ggml_tensor ref_mul = *p.mul;
    ggml_tensor ref_out = *p.out;
    ref_norm.data = scratch.get();
    ref_mul.data = scratch.get() + n;
    ref_mul.src[0] = &ref_norm;
    ref_out.data = scratch.get() + 2 * n;
    ref_out.src[0] = &ref_mul;
    ref_out.src[1] = &ref_norm;
    ggml_cuda_op_rms_norm(ctx, &ref_norm);
    ggml_cuda_op_mul(ctx, &ref_mul);
    ggml_cuda_op_add(ctx, &ref_out);
    launch(p, scratch.get() + 3 * n, stream);
    std::vector<float> base(n), cand(n);
    CUDA_CHECK(cudaMemcpyAsync(base.data(), ref_out.data, n * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(cand.data(), scratch.get() + 3 * n, n * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    double maximum = 0.0, squares = 0.0, base_squares = 0.0;
    size_t nonfinite = 0, worst = 0;
    for (size_t j = 0; j < n; ++j) {
        if (!std::isfinite(base[j]) || !std::isfinite(cand[j])) {
            ++nonfinite;
            continue;
        }
        const double d = double(base[j]) - double(cand[j]);
        squares += d * d;
        base_squares += double(base[j]) * double(base[j]);
        if (std::abs(d) > maximum) {
            maximum = std::abs(d);
            worst = j;
        }
    }
    GGML_LOG_INFO("flashrt_rmsnorm_shadow: math=ggml node=%s rows=%lld dim=%d "
                  "nonfinite=%llu ab_max=%.9g ab_rmse=%.9g ab_rel_l2=%.9g "
                  "worst_row=%llu worst_col=%llu propagate=ggml\n",
                  p.out->name, (long long) p.rows, p.dim, (unsigned long long) nonfinite,
                  maximum, std::sqrt(squares / n),
                  std::sqrt(squares / std::max(base_squares, 1e-300)),
                  (unsigned long long) (worst / p.dim), (unsigned long long) (worst % p.dim));
}

} // namespace

bool ggml_cuda_flashrt_rmsnorm(ggml_backend_cuda_context & ctx, ggml_cgraph * graph, int i) {
    static const bool enabled = enabled_env("GGML_CUDA_FLASHRT") && enabled_env("GGML_CUDA_FLASHRT_RMSNORM");
    if (!enabled || ggml_cuda_info().devices[ctx.device].cc != GGML_CUDA_CC_ADA_LOVELACE) {
        return false;
    }
    ggml_flashrt_rmsnorm_plan plan;
    if (!ggml_flashrt_plan_rmsnorm(graph, i, plan)) {
        return false;
    }
    static const bool do_shadow = enabled_env("GGML_CUDA_FLASHRT_SHADOW");
    if (do_shadow) {
        shadow(ctx, plan);
        return false;
    }
    launch(plan, static_cast<float *>(plan.out->data), ctx.stream());
    static const bool trace = enabled_env("GGML_CUDA_FLASHRT_TRACE");
    if (trace) {
        GGML_LOG_INFO("flashrt_rmsnorm: enqueued rows=%lld dim=%d node=%s math=ggml\n",
                      (long long) plan.rows, plan.dim, plan.out->name);
    }
    return true;
}
#endif
