// SPDX-License-Identifier: Apache-2.0
// Adapted from FlashRT csrc/kernels/norm.cu (layer_norm_kernel) and
// csrc/kernels/common.cuh (warp/block reductions), commit
// e881e1424cf4fadee7dff121d01cc4e159e550e7.
// Modifications: FP32/float2 IO, ggml graph dispatch and alias checks,
// explicit rounding of the affine epilogue, SM89-only opt-in integration.
// See vendor/flashrt/LICENSE and vendor/flashrt/README.md.

#ifdef GGML_CUDA_FLASHRT
#include "flashrt-layernorm.cuh"
#include "binbcast.cuh"
#include "norm.cuh"
#include "ggml-impl.h"

#include <algorithm>
#include <cmath>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace {

__device__ __forceinline__ float flashrt_warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__device__ __forceinline__ float flashrt_block_sum(float value, float * shared) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    // Protect the previous reduction's shared[0] readers on repeated calls.
    __syncthreads();
    value = flashrt_warp_sum(value);
    if (lane == 0) {
        shared[warp] = value;
    }
    __syncthreads();
    value = threadIdx.x < 8 ? shared[threadIdx.x] : 0.0f;
    if (warp == 0) {
        value = flashrt_warp_sum(value);
    }
    if (threadIdx.x == 0) {
        shared[0] = value;
    }
    __syncthreads();
    return shared[0];
}

// FlashRT's two-pass variance and packed-pair affine normalization,
// adapted to ggml's FP32 activation contract. No FP16 round trip.
__global__ void flashrt_layer_norm_affine_f32(
        const float2 * x, const float2 * weight, const float2 * bias,
        float2 * out, int dim, float eps) {
    const int pairs = dim / 2;
    x += int64_t(blockIdx.x) * pairs;
    out += int64_t(blockIdx.x) * pairs;
    __shared__ float shared[8];

    float sum = 0.0f;
    for (int col = threadIdx.x; col < pairs; col += blockDim.x) {
        const float2 value = x[col];
        sum += value.x + value.y;
    }
    const float mean = flashrt_block_sum(sum, shared) / dim;
    float variance = 0.0f;
    for (int col = threadIdx.x; col < pairs; col += blockDim.x) {
        const float2 value = x[col];
        const float dx = value.x - mean;
        const float dy = value.y - mean;
        variance += dx * dx + dy * dy;
    }
    const float inv_std = rsqrtf(flashrt_block_sum(variance, shared) / dim + eps);
    for (int col = threadIdx.x; col < pairs; col += blockDim.x) {
        const float2 value = x[col];
        const float2 w = weight[col];
        const float2 b = bias[col];
        // Preserve separate FP32 NORM/MUL/ADD rounding at the epilogue.
        const float nx = __fmul_rn(value.x - mean, inv_std);
        const float ny = __fmul_rn(value.y - mean, inv_std);
        out[col] = make_float2(__fadd_rn(__fmul_rn(nx, w.x), b.x),
                              __fadd_rn(__fmul_rn(ny, w.y), b.y));
    }
}

// Diagnostic control based on ggml-cuda/norm.cu, not the FlashRT reduction.
// Keep ggml's column assignment, float2 sum/sumsq reduction, variance expression
// and normalization expression. Only fuse the subsequent FP32 MUL and ADD.
// Matching source arithmetic is intentional; GPU tests must establish parity.
template <int block_size>
__global__ void flashrt_layer_norm_affine_ggml_math_f32(
        const float * x, const float * weight, const float * bias,
        float * dst, int ncols, float eps) {
    x   += int64_t(blockIdx.x) * ncols;
    dst += int64_t(blockIdx.x) * ncols;
    const int tid = threadIdx.x;

    float2 mean_var = make_float2(0.0f, 0.0f);
    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        mean_var.x += xi;
        mean_var.y += xi * xi;
    }

    extern __shared__ float2 s_sum2[];
    mean_var = block_reduce<block_reduce_method::SUM, block_size>(mean_var, s_sum2);
    const float mean = mean_var.x / ncols;
    const float var = mean_var.y / ncols - mean * mean;
    const float inv_std = rsqrtf(var + eps);

    for (int col = tid; col < ncols; col += block_size) {
        const float normalized = (x[col] - mean) * inv_std;
        dst[col] = __fadd_rn(__fmul_rn(normalized, weight[col]), bias[col]);
    }
}

bool flashrt_use_ggml_math() {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_CUDA_FLASHRT_GGML_MATH");
        return value && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

void flashrt_launch_layernorm(const float * x, const float * w, const float * b,
        float * out, int dim, int64_t rows, float eps, cudaStream_t stream) {
    if (flashrt_use_ggml_math()) {
        if (dim < 1024) {
            flashrt_layer_norm_affine_ggml_math_f32<WARP_SIZE><<<int(rows), WARP_SIZE, 0, stream>>>(
                x, w, b, out, dim, eps);
        } else {
            flashrt_layer_norm_affine_ggml_math_f32<1024><<<int(rows), 1024, 32 * sizeof(float2), stream>>>(
                x, w, b, out, dim, eps);
        }
    } else {
        flashrt_layer_norm_affine_f32<<<int(rows), 256, 0, stream>>>(
            reinterpret_cast<const float2 *>(x), reinterpret_cast<const float2 *>(w),
            reinterpret_cast<const float2 *>(b), reinterpret_cast<float2 *>(out), dim, eps);
    }
    CUDA_CHECK(cudaGetLastError());
}

bool contiguous_f32(const ggml_tensor * t) {
    return t && t->data && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t);
}

bool aligned_pair(const ggml_tensor * t) {
    return uintptr_t(t->data) % alignof(float2) == 0;
}

bool overlaps(const ggml_tensor * a, const ggml_tensor * b) {
    const uintptr_t x = uintptr_t(a->data);
    const uintptr_t y = uintptr_t(b->data);
    return x <= y ? y - x < ggml_nbytes(a) : x - y < ggml_nbytes(b);
}

// Diagnostic only. Neither calculation below writes a graph tensor. Returning
// false from the caller leaves the entire original graph execution in place.
void flashrt_layernorm_shadow(ggml_backend_cuda_context & ctx,
        const ggml_tensor * norm, const ggml_tensor * mul, const ggml_tensor * out,
        int dim, int64_t rows, float eps) {
    // This path allocates and synchronizes. Never run it during graph capture.
    if (std::getenv("GGML_CUDA_DISABLE_GRAPHS") == nullptr) {
        GGML_ABORT("FlashRT shadow requires GGML_CUDA_DISABLE_GRAPHS=1; restart the process");
    }
    const cudaStream_t stream = ctx.stream();
    cudaStreamCaptureStatus capture_status;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
    GGML_ASSERT(capture_status == cudaStreamCaptureStatusNone);

    const ggml_tensor * x = norm->src[0];
    const ggml_tensor * w = mul->src[1];
    const ggml_tensor * b = out->src[1];
    const size_t n = size_t(rows) * size_t(dim);
    ggml_cuda_pool_alloc<float> scratch(ctx.pool(), 4 * n);

    // Copies of descriptors only; keep real x/w/b and redirect all destinations.
    ggml_tensor ref_norm = *norm;
    ggml_tensor ref_mul = *mul;
    ggml_tensor ref_out = *out;
    ref_norm.data = scratch.get();
    ref_mul.data = scratch.get() + n;
    ref_out.data = scratch.get() + 2 * n;
    ref_norm.buffer = ref_mul.buffer = ref_out.buffer = nullptr;
    ref_norm.view_src = ref_mul.view_src = ref_out.view_src = nullptr;
    ref_norm.view_offs = ref_mul.view_offs = ref_out.view_offs = 0;
    ref_mul.src[0] = &ref_norm;
    ref_out.src[0] = &ref_mul;
    float * candidate = scratch.get() + 3 * n;

    ggml_cuda_op_norm(ctx, &ref_norm);
    ggml_cuda_op_mul(ctx, &ref_mul);
    ggml_cuda_op_add(ctx, &ref_out);
    CUDA_CHECK(cudaGetLastError());
    flashrt_launch_layernorm(static_cast<const float *>(x->data), static_cast<const float *>(w->data),
        static_cast<const float *>(b->data), candidate, dim, rows, eps, stream);

    std::vector<float> hx(n), hw(dim), hb(dim), hbase(n), hcand(n);
    CUDA_CHECK(cudaMemcpyAsync(hx.data(), x->data, n * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(hw.data(), w->data, dim * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(hb.data(), b->data, dim * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(hbase.data(), ref_out.data, n * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(hcand.data(), candidate, n * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    double ab_max = 0, ab_sq = 0, base_sq = 0, base_fp64_max = 0, cand_fp64_max = 0;
    double min_var = std::numeric_limits<double>::infinity(), max_abs_mean = 0;
    size_t worst = 0, nonfinite = 0;
    for (int64_t row = 0; row < rows; ++row) {
        const size_t start = size_t(row) * size_t(dim);
        double mean = 0, variance = 0;
        for (int col = 0; col < dim; ++col) {
            mean += double(hx[start + col]);
        }
        mean /= dim;
        for (int col = 0; col < dim; ++col) {
            const double d = double(hx[start + col]) - mean;
            variance += d * d;
        }
        variance /= dim;
        min_var = std::min(min_var, variance);
        max_abs_mean = std::max(max_abs_mean, std::abs(mean));
        const double scale = 1.0 / std::sqrt(variance + double(eps));
        for (int col = 0; col < dim; ++col) {
            const size_t j = start + col;
            const double expected = (double(hx[j]) - mean) * scale * double(hw[col]) + double(hb[col]);
            if (!std::isfinite(hbase[j]) || !std::isfinite(hcand[j]) || !std::isfinite(expected)) {
                ++nonfinite;
                continue;
            }
            const double d = double(hcand[j]) - double(hbase[j]);
            if (std::abs(d) > ab_max) {
                ab_max = std::abs(d);
                worst = j;
            }
            ab_sq += d * d;
            base_sq += double(hbase[j]) * double(hbase[j]);
            base_fp64_max = std::max(base_fp64_max, std::abs(double(hbase[j]) - expected));
            cand_fp64_max = std::max(cand_fp64_max, std::abs(double(hcand[j]) - expected));
        }
    }
    const double ab_rmse = std::sqrt(ab_sq / double(n));
    const double ab_rel_l2 = std::sqrt(ab_sq / std::max(base_sq, 1e-300));
    // Any nonfinite makes the finite-subset error metrics unusable as a pass.
    GGML_LOG_INFO("flashrt_shadow: math=%s node=%s rows=%lld dim=%d eps=%.9g nonfinite=%llu "
                  "ab_max=%.9g ab_rmse=%.9g ab_rel_l2=%.9g "
                  "base_fp64_max=%.9g cand_fp64_max=%.9g min_var=%.9g max_abs_mean=%.9g "
                  "worst_row=%llu worst_col=%llu base=%.9g cand=%.9g propagate=ggml\n",
                  flashrt_use_ggml_math() ? "ggml" : "flashrt", out->name,
                  (long long) rows, dim, double(eps), (unsigned long long) nonfinite,
                  ab_max, ab_rmse, ab_rel_l2, base_fp64_max, cand_fp64_max, min_var, max_abs_mean,
                  (unsigned long long) (worst / size_t(dim)), (unsigned long long) (worst % size_t(dim)),
                  double(hbase[worst]), double(hcand[worst]));
}

} // namespace

bool ggml_cuda_flashrt_layernorm(ggml_backend_cuda_context & ctx, ggml_cgraph * graph, int i) {
    // Read once, before graph capture. A/B runs must use separate processes.
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_CUDA_FLASHRT");
        return value && std::strcmp(value, "1") == 0;
    }();
    if (!enabled || ggml_cuda_info().devices[ctx.device].cc != GGML_CUDA_CC_ADA_LOVELACE ||
        !ggml_can_fuse(graph, i, { GGML_OP_NORM, GGML_OP_MUL, GGML_OP_ADD })) {
        return false;
    }

    const ggml_tensor * norm = graph->nodes[i];
    const ggml_tensor * mul = graph->nodes[i + 1];
    ggml_tensor * out = graph->nodes[i + 2];
    const ggml_tensor * x = norm->src[0];
    const ggml_tensor * w = mul->src[1];
    const ggml_tensor * b = out->src[1];
    // Only the canonical affine chain, with channel vectors broadcast over rows.
    if (mul->src[0] != norm || out->src[0] != mul ||
        !contiguous_f32(x) || !contiguous_f32(norm) || !contiguous_f32(mul) ||
        !contiguous_f32(out) || !contiguous_f32(w) || !contiguous_f32(b) ||
        !ggml_are_same_shape(x, out) || !ggml_are_same_shape(x, norm) ||
        !ggml_are_same_shape(x, mul)) {
        return false;
    }
    const int64_t dim = x->ne[0];
    const int64_t rows = ggml_nrows(x);
    if (dim < 2 || dim > 8192 || dim % 2 != 0 || rows < 1 || rows > INT_MAX ||
        w->ne[0] != dim || ggml_nelements(w) != dim ||
        b->ne[0] != dim || ggml_nelements(b) != dim ||
        !aligned_pair(x) || !aligned_pair(w) || !aligned_pair(b) || !aligned_pair(out) ||
        overlaps(out, x) || overlaps(out, w) || overlaps(out, b)) {
        return false;
    }
    float eps;
    std::memcpy(&eps, norm->op_params, sizeof(eps));
    if (!std::isfinite(eps) || eps <= 0.0f) {
        return false;
    }

    static const bool shadow = [] {
        const char * value = std::getenv("GGML_CUDA_FLASHRT_SHADOW");
        return value && std::strcmp(value, "1") == 0;
    }();
    if (shadow) {
        flashrt_layernorm_shadow(ctx, norm, mul, out, int(dim), rows, eps);
        return false; // let ggml execute the original graph, including its other fusions
    }

    flashrt_launch_layernorm(static_cast<const float *>(x->data), static_cast<const float *>(w->data),
        static_cast<const float *>(b->data), static_cast<float *>(out->data), int(dim), rows, eps, ctx.stream());
    static const bool trace = [] {
        const char * value = std::getenv("GGML_CUDA_FLASHRT_TRACE");
        return value && std::strcmp(value, "1") == 0;
    }();
    if (trace) {
        GGML_LOG_INFO("flashrt_layernorm: enqueued rows=%lld dim=%lld node=%s math=%s\n",
                      (long long) rows, (long long) dim, out->name,
                      flashrt_use_ggml_math() ? "ggml" : "flashrt");
    }
    return true;
}
#endif // GGML_CUDA_FLASHRT
