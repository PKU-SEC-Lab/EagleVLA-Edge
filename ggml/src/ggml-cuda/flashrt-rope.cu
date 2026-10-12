// SPDX-License-Identifier: Apache-2.0
// Rotation adapted from FlashRT qkv_split_rope_kvcache_fp16_kernel, in
// csrc/kernels/rope.cu at e881e1424cf4fadee7dff121d01cc4e159e550e7:
// https://github.com/flashrt-project/FlashRT/blob/e881e1424cf4fadee7dff121d01cc4e159e550e7/csrc/kernels/rope.cu
// Coefficients follow flash_rt/models/pi0/pipeline_rtx.py's FP64 table builder.
// Complete upstream license: vendor/flashrt/rope/LICENSE.
//
// Retains FP16 inputs, coefficients and rounded outputs, and the native
// per-output-scalar FP32 rotation expressions (including compiler FMA choice).
// Adaptations: GGML F32 boundary, NEOX split-half pairs instead of native
// adjacent pairs, separate Q/K tensors, GPU position vector and persistent
// per-context coefficients. Does NOT port packed QKV splitting or KV writes.

#ifdef GGML_CUDA_FLASHRT

#include "flashrt-rope.cuh"
#include "flashrt-rope-guard.h"
#include "common.cuh"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <new>

namespace {

static constexpr int rope_pairs = GGML_FLASHRT_ROPE_HEAD_DIM / 2;
static constexpr size_t rope_table_count = size_t(GGML_FLASHRT_ROPE_LUT_POSITIONS) * rope_pairs;
static constexpr size_t rope_table_bytes = rope_table_count * sizeof(half2);
static constexpr size_t rope_cache_bytes = rope_table_bytes + rope_pairs * sizeof(double);
static_assert(rope_table_bytes % alignof(double) == 0, "inverse frequency alignment");

struct flashrt_rope_cache {
    int device;
    void * allocation;
};

static bool rope_env_one(const char * name) {
    const char * value = std::getenv(name);
    return value && std::strcmp(value, "1") == 0;
}

static void rope_cache_delete(void * pointer) {
    flashrt_rope_cache * cache = static_cast<flashrt_rope_cache *>(pointer);
    if (!cache) { return; }
    int previous = -1;
    const cudaError_t get_error = cudaGetDevice(&previous);
    const cudaError_t select_error = cudaSetDevice(cache->device);
    if (select_error == cudaSuccess) {
        // This is ordinary cudaMalloc memory. cudaFree waits for outstanding
        // accesses. Context member order destroys captured graph objects before
        // releasing this allocation, even when backend creation fails later.
        const cudaError_t free_error = cudaFree(cache->allocation);
        if (free_error != cudaSuccess) {
            GGML_LOG_WARN("flashrt_rope_init: cache cleanup failed: %s\n", cudaGetErrorString(free_error));
        }
    } else {
        GGML_LOG_WARN("flashrt_rope_init: cannot select device for cache cleanup: %s\n",
            cudaGetErrorString(select_error));
    }
    if (get_error == cudaSuccess && previous != cache->device) {
        const cudaError_t restore_error = cudaSetDevice(previous);
        if (restore_error != cudaSuccess) {
            GGML_LOG_WARN("flashrt_rope_init: cannot restore device after cleanup: %s\n",
                cudaGetErrorString(restore_error));
        }
    }
    delete cache;
}

static bool rope_on_device(const ggml_tensor * tensor, size_t bytes, int device) {
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (!buffer || (tensor->buffer && tensor->buffer != buffer) ||
        ggml_backend_buffer_get_type(buffer) != ggml_backend_cuda_buffer_type(device)) {
        return false;
    }
    const uintptr_t base = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(buffer));
    const uintptr_t begin = reinterpret_cast<uintptr_t>(tensor->data);
    const size_t size = ggml_backend_buffer_get_size(buffer);
    return base != 0 && begin >= base && begin - base <= size && bytes <= size - size_t(begin - base);
}

static bool rope_reject(ggml_tensor * dst, const char * reason, bool trace) {
    if (trace && dst) {
        GGML_LOG_INFO("flashrt_rope_skip: node=%s reason=%s\n", dst->name, reason);
    }
    return false;
}

static __global__ void flashrt_rope_rotation_fp16_kernel(
        const float * __restrict__ input, const int32_t * __restrict__ positions,
        float * __restrict__ output, const half2 * __restrict__ table,
        const double * __restrict__ inverse_frequency, int heads, int rows) {
    const int index = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    const int elements = GGML_FLASHRT_ROPE_HEAD_DIM * heads * rows;
    if (index >= elements) { return; }
    const int component = index % GGML_FLASHRT_ROPE_HEAD_DIM;
    const int pair = component % rope_pairs;
    const int head_base = index - component;
    const int token = index / (GGML_FLASHRT_ROPE_HEAD_DIM * heads);
    const int32_t position = positions[token];

    half2 coefficient;
    if (position >= 0 && position < GGML_FLASHRT_ROPE_LUT_POSITIONS) {
        coefficient = table[size_t(position) * rope_pairs + pair];
    } else {
        // No position-dependent host copy or capture-time allocation. A changed
        // position vector is read on every graph replay. Negative and extreme
        // int32 values never index the finite LUT; the angle stays finite.
        const double angle = double(position) * inverse_frequency[pair];
        coefficient = __halves2half2(__double2half(cos(angle)), __double2half(sin(angle)));
    }
    const float cos_v = __half2float(__low2half(coefficient));
    const float sin_v = __half2float(__high2half(coefficient));
    const float x0 = __half2float(__float2half_rn(input[head_base + pair]));
    const float x1 = __half2float(__float2half_rn(input[head_base + pair + rope_pairs]));
    const half rotated = component < rope_pairs ? __float2half_rn(x0 * cos_v - x1 * sin_v)
                                                : __float2half_rn(x1 * cos_v + x0 * sin_v);
    output[index] = __half2float(rotated);
}

} // namespace

void ggml_cuda_flashrt_rope_init(ggml_backend_cuda_context & ctx) {
    if (ctx.flashrt_rope_state || !rope_env_one("GGML_CUDA_FLASHRT") ||
        !rope_env_one("GGML_CUDA_FLASHRT_ROPE") ||
        ggml_cuda_info().devices[ctx.device].cc != GGML_CUDA_CC_ADA_LOVELACE) {
        return;
    }
    // Initialization is called from backend creation, never from op dispatch.
    // All storage and coefficient transfers finish before any graph capture.
    // Do not mistake a pre-existing error for this optional allocation failure.
    CUDA_CHECK(cudaGetLastError());
    int previous = -1;
    const cudaError_t previous_error = cudaGetDevice(&previous);
    CUDA_CHECK(previous_error);
    cudaError_t error = cudaSetDevice(ctx.device);
    CUDA_CHECK(error);
    try {
        std::vector<half2> table(rope_table_count);
        std::array<double, rope_pairs> inverse_frequency{};
        for (int d = 0; d < rope_pairs; ++d) {
            // Match the host pipeline's inverse-power expression, preserving
            // FP64 through trig and the final direct FP64-to-FP16 RN conversion.
            // CUDA's host/device __double2half is directly RN-even; an extra
            // double-to-float step would introduce double-rounding artifacts.
            inverse_frequency[d] = 1.0 / std::pow(10000.0, (2.0 * d) / GGML_FLASHRT_ROPE_HEAD_DIM);
        }
        for (int position = 0; position < GGML_FLASHRT_ROPE_LUT_POSITIONS; ++position) {
            for (int d = 0; d < rope_pairs; ++d) {
                const double angle = double(position) * inverse_frequency[d];
                table[size_t(position) * rope_pairs + d] =
                    __halves2half2(__double2half(std::cos(angle)), __double2half(std::sin(angle)));
            }
        }
        std::shared_ptr<void> state(new flashrt_rope_cache{ctx.device, nullptr}, rope_cache_delete);
        flashrt_rope_cache * cache = static_cast<flashrt_rope_cache *>(state.get());
        error = cudaMalloc(&cache->allocation, rope_cache_bytes);
        if (error == cudaErrorMemoryAllocation) {
            // This is the only recoverable CUDA initialization error. Clear
            // that precise last-error code so a later GGML launch cannot
            // inherit it. A different pending code is not ours to suppress.
            const cudaError_t pending = cudaGetLastError();
            if (pending != cudaSuccess && pending != cudaErrorMemoryAllocation) {
                CUDA_CHECK(pending);
            }
        } else {
            CUDA_CHECK(error);
        }
        if (error == cudaSuccess) {
            error = cudaMemcpy(cache->allocation, table.data(), rope_table_bytes, cudaMemcpyHostToDevice);
            CUDA_CHECK(error);
        }
        if (error == cudaSuccess) {
            error = cudaMemcpy(static_cast<char *>(cache->allocation) + rope_table_bytes,
                inverse_frequency.data(), sizeof(inverse_frequency), cudaMemcpyHostToDevice);
            CUDA_CHECK(error);
        }
        if (error == cudaSuccess) {
            // Make completion explicit before subsequently-created nonblocking
            // GGML streams or a graph capture can consume these coefficients.
            error = cudaStreamSynchronize(nullptr);
            CUDA_CHECK(error);
        }
        if (error == cudaSuccess) {
            ctx.flashrt_rope_state = std::move(state);
            if (rope_env_one("GGML_CUDA_FLASHRT_TRACE")) {
                GGML_LOG_INFO("flashrt_rope_init: device=%d lut_positions=%d head_dim=%d cache_bytes=%zu\n",
                    ctx.device, GGML_FLASHRT_ROPE_LUT_POSITIONS, GGML_FLASHRT_ROPE_HEAD_DIM, rope_cache_bytes);
            }
        } else {
            GGML_LOG_WARN("flashrt_rope_init: cache unavailable, using GGML: %s\n", cudaGetErrorString(error));
        }
    } catch (const std::bad_alloc &) {
        GGML_LOG_WARN("flashrt_rope_init: host allocation unavailable, using GGML\n");
    }
    if (previous_error == cudaSuccess && previous != ctx.device) {
        CUDA_CHECK(cudaSetDevice(previous));
    }
}

bool ggml_cuda_flashrt_rope(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    static const bool enabled = rope_env_one("GGML_CUDA_FLASHRT") && rope_env_one("GGML_CUDA_FLASHRT_ROPE");
    static const bool trace = rope_env_one("GGML_CUDA_FLASHRT_TRACE");
    if (!enabled) { return false; }
    if (ggml_cuda_info().devices[ctx.device].cc != GGML_CUDA_CC_ADA_LOVELACE) {
        return rope_reject(dst, "device_architecture", trace);
    }
    if (!ctx.flashrt_rope_state) { return rope_reject(dst, "coefficient_cache_unavailable", trace); }
    ggml_flashrt_rope_plan plan{};
    const char * rejection = nullptr;
    if (!ggml_flashrt_plan_rope(dst, plan, &rejection)) { return rope_reject(dst, rejection, trace); }
    if (!rope_on_device(plan.src, plan.tensor_bytes, ctx.device) ||
        !rope_on_device(plan.positions, plan.position_bytes, ctx.device) ||
        !rope_on_device(plan.dst, plan.tensor_bytes, ctx.device)) {
        return rope_reject(dst, "device_buffer_or_range", trace);
    }
    const flashrt_rope_cache * cache = static_cast<const flashrt_rope_cache *>(ctx.flashrt_rope_state.get());
    cudaStream_t stream = ctx.stream();
    const int elements = GGML_FLASHRT_ROPE_HEAD_DIM * plan.heads * plan.rows;
    flashrt_rope_rotation_fp16_kernel<<<(elements + 255) / 256, 256, 0, stream>>>(
        static_cast<const float *>(plan.src->data), static_cast<const int32_t *>(plan.positions->data),
        static_cast<float *>(plan.dst->data), static_cast<const half2 *>(cache->allocation),
        reinterpret_cast<const double *>(static_cast<const char *>(cache->allocation) + rope_table_bytes),
        plan.heads, plan.rows);
    CUDA_CHECK(cudaGetLastError());
    if (trace) {
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
        GGML_LOG_INFO("flashrt_rope: node=%s heads=%d rows=%d input=f32 output=f32 math=fp16 layout=neox capture=%d\n",
            dst->name, plan.heads, plan.rows, capture == cudaStreamCaptureStatusActive ? 1 : 0);
    }
    return true;
}

#endif // GGML_CUDA_FLASHRT
