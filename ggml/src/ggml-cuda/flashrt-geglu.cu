// SPDX-License-Identifier: Apache-2.0
// Adapted from FlashRT csrc/kernels/activation.cu: gate_geglu_kernel<T>
// and gate_geglu_merged_kernel<T>, pinned commit
// e881e1424cf4fadee7dff121d01cc4e159e550e7:
// https://github.com/flashrt-project/FlashRT/blob/e881e1424cf4fadee7dff121d01cc4e159e550e7/csrc/kernels/activation.cu
// The complete upstream license is in vendor/flashrt/geglu/LICENSE.
//
// Changes: GGML split/merged row strides and swapped-half addressing, explicit
// SM89 dispatch, address/layout guards and optional graph-capture trace.
// FP16 retains the upstream input/output dtype and FP32 scalar arithmetic;
// F32 is a dtype extension for existing GGML MUL_MAT outputs, without an FP16
// round trip. This preserves the upstream expf GEGLU expression. It does NOT
// substitute GGML's tanhf expression and does not claim bitwise parity with it.
// No FP8, amax, quantization or weight transformation is implemented here.

#ifdef GGML_CUDA_FLASHRT

#include "flashrt-geglu.cuh"
#include "flashrt-geglu-guard.h"
#include "common.cuh"

#include <cstdlib>
#include <cstring>

namespace {

static bool flashrt_geglu_env_one(const char * name) {
    const char * value = std::getenv(name);
    return value && std::strcmp(value, "1") == 0;
}

static bool flashrt_geglu_device_tensor(const ggml_tensor * t, int device) {
    return t && t->buffer &&
        ggml_backend_buffer_get_type(t->buffer) == ggml_backend_cuda_buffer_type(device);
}

template <typename T>
static __device__ __forceinline__ float flashrt_geglu_to_f32(T value) {
    return static_cast<float>(value);
}

template <typename T>
static __device__ __forceinline__ T flashrt_geglu_from_f32(float value) {
    return static_cast<T>(value);
}

template <>
__device__ __forceinline__ half flashrt_geglu_from_f32<half>(float value) {
    return __float2half_rn(value);
}

// The packed split instantiation is the upstream one-element-per-thread work
// assignment. The other instantiation additionally maps a logical flat index
// into the two GGML row strides (also used for merged and swapped inputs).
// Gate and up may alias each other as read-only inputs; dst is disjoint from
// both by contract. No scratch buffers, synchronizations or graph edits.
template <typename T, bool packed_split>
static __global__ void flashrt_gate_geglu_kernel(
        const T * gate, const T * up, T * output, int elements, int width,
        size_t gate_stride, size_t up_stride, size_t gate_offset, size_t up_offset) {
    const int index = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (index >= elements) {
        return;
    }
    size_t gate_index = size_t(index);
    size_t up_index = size_t(index);
    if (!packed_split) {
        const int row = index / width;
        const int col = index % width;
        gate_index = size_t(row) * gate_stride + size_t(col) + gate_offset;
        up_index = size_t(row) * up_stride + size_t(col) + up_offset;
    }
    const float g = flashrt_geglu_to_f32(gate[gate_index]);
    const float u = flashrt_geglu_to_f32(up[up_index]);
    const float gelu = g / (1.0f + expf(-1.5957691216057308f * g * (1.0f + 0.044715f * g * g)));
    output[index] = flashrt_geglu_from_f32<T>(gelu * u);
}

template <typename T>
static void flashrt_geglu_launch(const ggml_flashrt_geglu_plan & plan, cudaStream_t stream) {
    const int blocks = (plan.elements + 255) / 256;
    const T * gate = static_cast<const T *>(plan.gate->data);
    const T * up = static_cast<const T *>(plan.up->data);
    T * output = static_cast<T *>(plan.output->data);
    const bool packed_split = !plan.merged &&
        plan.gate_row_stride == size_t(plan.width) && plan.up_row_stride == size_t(plan.width);
    if (packed_split) {
        flashrt_gate_geglu_kernel<T, true><<<blocks, 256, 0, stream>>>(
            gate, up, output, plan.elements, plan.width, plan.gate_row_stride,
            plan.up_row_stride, plan.gate_offset, plan.up_offset);
    } else {
        flashrt_gate_geglu_kernel<T, false><<<blocks, 256, 0, stream>>>(
            gate, up, output, plan.elements, plan.width, plan.gate_row_stride,
            plan.up_row_stride, plan.gate_offset, plan.up_offset);
    }
}

} // namespace

bool ggml_cuda_flashrt_geglu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    static const bool enabled = flashrt_geglu_env_one("GGML_CUDA_FLASHRT") &&
        flashrt_geglu_env_one("GGML_CUDA_FLASHRT_GEGLU");
    static const bool trace = flashrt_geglu_env_one("GGML_CUDA_FLASHRT_TRACE");
    if (!enabled || ggml_cuda_info().devices[ctx.device].cc != GGML_CUDA_CC_ADA_LOVELACE) {
        return false;
    }
    ggml_flashrt_geglu_plan plan{};
    const char * rejection = nullptr;
    if (!ggml_flashrt_plan_geglu(dst, plan, trace ? &rejection : nullptr)) {
        if (trace && dst) {
            GGML_LOG_INFO("flashrt_geglu_skip: node=%s reason=%s\n", dst->name, rejection);
        }
        return false;
    }
    if (!flashrt_geglu_device_tensor(plan.gate, ctx.device) ||
        !flashrt_geglu_device_tensor(plan.up, ctx.device) ||
        !flashrt_geglu_device_tensor(plan.output, ctx.device)) {
        if (trace) {
            GGML_LOG_INFO("flashrt_geglu_skip: node=%s reason=device_buffer\n", dst->name);
        }
        return false;
    }
    cudaStream_t stream = ctx.stream();
    if (dst->type == GGML_TYPE_F16) {
        flashrt_geglu_launch<half>(plan, stream);
    } else {
        flashrt_geglu_launch<float>(plan, stream);
    }
    CUDA_CHECK(cudaGetLastError());
    if (trace) {
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
        GGML_LOG_INFO("flashrt_geglu: math=flashrt dtype=%s layout=%s swapped=%d width=%d rows=%d capture=%d node=%s\n",
            dst->type == GGML_TYPE_F16 ? "f16" : "f32", plan.merged ? "merged" : "split",
            plan.swapped ? 1 : 0, plan.width, plan.rows,
            capture == cudaStreamCaptureStatusActive ? 1 : 0, dst->name);
    }
    return true;
}

#endif // GGML_CUDA_FLASHRT
