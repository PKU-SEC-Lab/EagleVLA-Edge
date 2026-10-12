// SPDX-License-Identifier: Apache-2.0
// Adapted from FlashRT patch_im2col_kernel and patch_im2col, in
// csrc/kernels/patch_embed.cu at e881e1424cf4fadee7dff121d01cc4e159e550e7:
// https://github.com/flashrt-project/FlashRT/blob/e881e1424cf4fadee7dff121d01cc4e159e550e7/csrc/kernels/patch_embed.cu
// Complete upstream license: vendor/flashrt/patch-im2col/LICENSE.
//
// Retains upstream's flat output index and one-element-per-thread gather.
// Changes: GGML NCHW image indexing and C/KH/KW patch-feature order, F32 input
// with F32 copy or F16 RN conversion, exact shape/device/range guards and an
// optional graph-capture trace. This adapts the gather to existing GGML image
// preprocessing and convolution weights, without repacking either. It is not
// the original NHWC FP16 image pipeline or the uint8/LUT preprocessing path.

#ifdef GGML_CUDA_FLASHRT

#include "flashrt-patch-im2col.cuh"
#include "flashrt-patch-im2col-guard.h"
#include "common.cuh"

#include <cstdlib>
#include <cstring>

namespace {

static bool patch_im2col_env_one(const char * name) {
    const char * value = std::getenv(name);
    return value && std::strcmp(value, "1") == 0;
}

static bool patch_im2col_on_device(const ggml_tensor * tensor, int device) {
    // GGML uses a view's source buffer as its owning allocation. Initialized
    // views normally also have tensor->buffer set to that same allocation.
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (!buffer || (tensor->buffer && tensor->buffer != buffer) ||
        ggml_backend_buffer_get_type(buffer) != ggml_backend_cuda_buffer_type(device)) {
        return false;
    }
    const uintptr_t base = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(buffer));
    const uintptr_t begin = reinterpret_cast<uintptr_t>(tensor->data);
    const size_t size = ggml_backend_buffer_get_size(buffer);
    // The pure guard proves every stride and ne, and checks overflow of the
    // complete extent. nb[3] is ONE image's size; ne[3] may be two here.
    const size_t bytes = tensor->nb[3] * size_t(tensor->ne[3]);
    return base != 0 && begin >= base && begin - base <= size &&
        bytes <= size - size_t(begin - base);
}

static bool patch_im2col_reject(ggml_tensor * dst, const char * reason, bool trace) {
    if (trace && dst) {
        GGML_LOG_INFO("flashrt_patch_im2col_skip: node=%s reason=%s\n", dst->name, reason);
    }
    return false;
}

template <typename T>
static __device__ __forceinline__ T patch_im2col_cast(float value) {
    return value;
}

template <>
__device__ __forceinline__ half patch_im2col_cast<half>(float value) {
    return __float2half_rn(value);
}

template <typename T>
static __global__ void flashrt_patch_im2col_kernel(
        const float * __restrict__ input, T * __restrict__ output, int batches) {
    const int idx = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    const int total = batches * 256 * 588;
    if (idx >= total) { return; }

    const int patch_idx = idx / 588;
    const int feat_idx = idx % 588;
    const int batch = patch_idx / 256;
    const int local_patch = patch_idx % 256;
    const int ph = local_patch / 16;
    const int pw = local_patch % 16;

    // GGML weights flatten C, KH, KW, with KW varying fastest.
    const int c = feat_idx / (14 * 14);
    const int pxh = (feat_idx % (14 * 14)) / 14;
    const int pxw = feat_idx % 14;
    const int row = ph * 14 + pxh;
    const int col = pw * 14 + pxw;
    const int src = ((batch * 3 + c) * 224 + row) * 224 + col;
    output[idx] = patch_im2col_cast<T>(input[src]);
}

} // namespace

bool ggml_cuda_flashrt_patch_im2col(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    static const bool enabled = patch_im2col_env_one("GGML_CUDA_FLASHRT") &&
        patch_im2col_env_one("GGML_CUDA_FLASHRT_PATCH_IM2COL");
    static const bool trace = patch_im2col_env_one("GGML_CUDA_FLASHRT_TRACE");
    if (!enabled) { return false; }
    if (ggml_cuda_info().devices[ctx.device].cc != GGML_CUDA_CC_ADA_LOVELACE) {
        return patch_im2col_reject(dst, "device_architecture", trace);
    }
    ggml_flashrt_patch_im2col_plan plan{};
    const char * rejection = nullptr;
    if (!ggml_flashrt_plan_patch_im2col(dst, plan, &rejection)) {
        return patch_im2col_reject(dst, rejection, trace);
    }
    if (!patch_im2col_on_device(plan.image, ctx.device) ||
        !patch_im2col_on_device(plan.output, ctx.device)) {
        return patch_im2col_reject(dst, "device_buffer_or_range", trace);
    }
    cudaStream_t stream = ctx.stream();
    const int total = plan.batch * 256 * 588;
    const int blocks = (total + 255) / 256;
    const float * image = static_cast<const float *>(plan.image->data);
    if (dst->type == GGML_TYPE_F16) {
        flashrt_patch_im2col_kernel<half><<<blocks, 256, 0, stream>>>(
            image, static_cast<half *>(dst->data), plan.batch);
    } else {
        flashrt_patch_im2col_kernel<float><<<blocks, 256, 0, stream>>>(
            image, static_cast<float *>(dst->data), plan.batch);
    }
    CUDA_CHECK(cudaGetLastError());
    if (trace) {
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
        GGML_LOG_INFO("flashrt_patch_im2col: node=%s input=f32 output=%s batch=%d "
                      "width=224 height=224 patch=14 capture=%d\n",
            dst->name, dst->type == GGML_TYPE_F16 ? "f16" : "f32", plan.batch,
            capture == cudaStreamCaptureStatusActive ? 1 : 0);
    }
    return true;
}

#endif // GGML_CUDA_FLASHRT
