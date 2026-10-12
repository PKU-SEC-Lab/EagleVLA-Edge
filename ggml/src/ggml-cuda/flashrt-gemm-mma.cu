// Adapted from ggml/src/ggml-cuda/mmf.cuh and mma.cuh at Jetson-PI-Edge
// commit 1271fa933c3f81e1cb3a71a9f09f13bdefa4461b.
// This is a GGML-math hardware adaptation candidate in the FlashRT port
// experiment; it is not an upstream FlashRT native GEMM implementation.
//
// MIT License
//
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifdef GGML_CUDA_FLASHRT

#include "flashrt-gemm-mma.cuh"
#include "common.cuh"
#include "convert.cuh"
#include "mma.cuh"

#include <cstdint>
#include <cstring>
#include <limits>

namespace {

constexpr int token_count = 11;
constexpr int output_rows = 16;
constexpr int warp_count = 8;
constexpr int warp_width = 32;
constexpr int shared_k_stride = warp_width + 4;
constexpr int reduce_stride = warp_count * output_rows + 4;
constexpr size_t load_shared_bytes = warp_count * 16 * shared_k_stride * sizeof(half2);
constexpr size_t reduce_shared_bytes = 16 * reduce_stride * sizeof(float);
constexpr size_t shared_bytes = load_shared_bytes > reduce_shared_bytes ? load_shared_bytes : reduce_shared_bytes;

static bool env_equals(const char * name, const char * expected) {
    const char * value = std::getenv(name);
    return value != nullptr && std::strcmp(value, expected) == 0;
}

static bool supported_shape(int64_t k, int64_t n) {
    return (k == 1024 && (n == 256 || n == 2048 || n == 4096)) ||
           (k == 2048 && n == 1024) || (k == 4096 && n == 1024);
}

static bool contiguous_matrix(const ggml_tensor * tensor, ggml_type type, int device) {
    return tensor != nullptr && tensor->type == type && tensor->data != nullptr &&
           tensor->buffer != nullptr && tensor->ne[0] > 0 && tensor->ne[1] > 0 &&
           tensor->ne[2] == 1 && tensor->ne[3] == 1 && ggml_is_contiguous(tensor) &&
           ggml_backend_buffer_get_type(tensor->buffer) == ggml_backend_cuda_buffer_type(device);
}

static bool overlaps_or_invalid(const ggml_tensor * a, const ggml_tensor * b) {
    const uintptr_t aa = reinterpret_cast<uintptr_t>(a->data);
    const uintptr_t bb = reinterpret_cast<uintptr_t>(b->data);
    const size_t as = ggml_nbytes(a);
    const size_t bs = ggml_nbytes(b);
    const uintptr_t limit = std::numeric_limits<uintptr_t>::max();
    if (as == 0 || bs == 0 || as > limit - aa || bs > limit - bb) {
        return true;
    }
    return aa < bb + bs && bb < aa + as;
}

// D[M,N] = RN16(X[M,K]) * W[N,K]^T; D and every accumulator remain F32.
// Keep the original 8 independent strided-K accumulators. For warp w their
// K64 chunks start at 64*w + 512*t. Each chunk uses the same four K16 MMA
// operations as mul_mat_f<half2,32,11,8,false>. Only independent output rows
// are retiled: 16 rows per CTA instead of 32. GPU equality remains a test gate.
__launch_bounds__(warp_width * warp_count, 1)
static __global__ void flashrt_gemm_ggml_mma16_kernel(
        const half2 * __restrict__ weights,
        const float2 * __restrict__ activations,
        float * __restrict__ dst,
        const int k_half2, const int n) {
#if defined(TURING_MMA_AVAILABLE) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 890
    using namespace ggml_cuda_mma;
    using tile_A = tile<16, 8, half2>;
    using tile_B = tile<8, 8, half2>;
    using tile_C = tile<16, 8, float>;

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int row0 = blockIdx.x * output_rows;
    weights += row0 * k_half2;

    // Base and per-warp offsets are 16-byte aligned for ldmatrix.
    extern __shared__ __align__(16) unsigned char storage[];
    half2 * xy = reinterpret_cast<half2 *>(storage) + warp * (16 * shared_k_stride);
    tile_C accum[2];

    for (int col = warp * warp_width + lane; col < k_half2; col += warp_count * warp_width) {
        tile_A a[warp_width / tile_A::J];
#pragma unroll
        for (int i = 0; i < tile_A::I; ++i) {
            xy[i * shared_k_stride + lane] = weights[i * k_half2 + col];
        }
        // ldmatrix.sync is a weak memory operation: explicitly order the
        // producer stores and subsequent reuse under independent scheduling.
        __syncwarp();
#pragma unroll
        for (int k0 = 0; k0 < warp_width; k0 += tile_A::J) {
            load_ldmatrix(a[k0 / tile_A::J], xy + k0, shared_k_stride);
        }
        __syncwarp();

#pragma unroll
        for (int it_b = 0; it_b < 2; ++it_b) {
#pragma unroll
            for (int j0 = 0; j0 < tile_B::I; ++j0) {
                const int j = j0 + it_b * tile_B::I;
                const float2 value = j < token_count ? activations[j * k_half2 + col] : make_float2(0.0f, 0.0f);
                xy[j0 * shared_k_stride + lane] = ggml_cuda_cast<half2>(value);
            }
            __syncwarp();
#pragma unroll
            for (int k0 = 0; k0 < warp_width; k0 += tile_B::J) {
                tile_B b;
                load_ldmatrix(b, xy + k0, shared_k_stride);
                mma(accum[it_b], a[k0 / tile_B::J], b);
            }
            __syncwarp();
        }
    }

    __syncthreads();
    float * partial = reinterpret_cast<float *>(storage);
#pragma unroll
    for (int it_b = 0; it_b < 2; ++it_b) {
#pragma unroll
        for (int l = 0; l < tile_C::ne; ++l) {
            const int i = warp * output_rows + tile_C::get_i(l);
            const int j = it_b * tile_C::J + tile_C::get_j(l);
            partial[j * reduce_stride + i] = accum[it_b].x[l];
        }
    }
    __syncthreads();

    // All 32 lanes participate in every MMA and barrier. Only the independent
    // final stores use 16 lanes. Do not replace this ordered sum with a tree,
    // atomics, split-K library reduction, or a different number of warps.
#pragma unroll
    for (int j0 = 0; j0 < token_count; j0 += warp_count) {
        const int j = j0 + warp;
        if (j < token_count && lane < output_rows) {
            float sum = 0.0f;
#pragma unroll
            for (int w = 0; w < warp_count; ++w) {
                sum += partial[j * reduce_stride + w * output_rows + lane];
            }
            dst[j * n + row0 + lane] = sum;
        }
    }
#else
    GGML_UNUSED_VARS(weights, activations, dst, k_half2, n);
    NO_DEVICE_CODE;
#endif
}

} // namespace

bool ggml_cuda_flashrt_gemm_mma(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    // Each benchmark/server process chooses its mode before backend dispatch.
    static const bool enabled = env_equals("GGML_CUDA_FLASHRT", "1") && env_equals("GGML_CUDA_FLASHRT_GEMM", "2");
    static const bool trace = env_equals("GGML_CUDA_FLASHRT_TRACE", "1");
    if (!enabled ||
        dst == nullptr || dst->op != GGML_OP_MUL_MAT || dst->op_params[0] != GGML_PREC_DEFAULT ||
        ggml_cuda_info().devices[ctx.device].cc != GGML_CUDA_CC_ADA_LOVELACE ||
        ggml_cuda_info().devices[ctx.device].warp_size != warp_width) {
        return false;
    }

    const ggml_tensor * weights = dst->src[0];
    const ggml_tensor * activations = dst->src[1];
    if (!contiguous_matrix(weights, GGML_TYPE_F16, ctx.device) ||
        !contiguous_matrix(activations, GGML_TYPE_F32, ctx.device) ||
        !contiguous_matrix(dst, GGML_TYPE_F32, ctx.device)) {
        return false;
    }
    const int64_t k = weights->ne[0];
    const int64_t n = weights->ne[1];
    if (!supported_shape(k, n) || activations->ne[0] != k || activations->ne[1] != token_count ||
        dst->ne[0] != n || dst->ne[1] != token_count ||
        reinterpret_cast<uintptr_t>(weights->data) % alignof(half2) != 0 ||
        reinterpret_cast<uintptr_t>(activations->data) % alignof(float2) != 0 ||
        reinterpret_cast<uintptr_t>(dst->data) % alignof(float) != 0 ||
        overlaps_or_invalid(weights, activations) || overlaps_or_invalid(weights, dst) ||
        overlaps_or_invalid(activations, dst)) {
        return false;
    }

    flashrt_gemm_ggml_mma16_kernel<<<dim3(unsigned(n / output_rows)), dim3(warp_width, warp_count), shared_bytes, ctx.stream()>>>(
        static_cast<const half2 *>(weights->data), static_cast<const float2 *>(activations->data),
        static_cast<float *>(dst->data), int(k / 2), int(n));
    CUDA_CHECK(cudaGetLastError());
    if (trace) {
        // This counts host enqueues (including capture), not CUDA Graph replays.
        cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(ctx.stream(), &capture_status));
        GGML_LOG_INFO("flashrt_gemm: mode=ggml_mma16 M=%d N=%lld K=%lld node=%s weight=%s capture=%d\n",
                      token_count, (long long) n, (long long) k, dst->name, weights->name,
                      capture_status == cudaStreamCaptureStatusActive ? 1 : 0);
    }
    return true;
}

#endif // GGML_CUDA_FLASHRT
