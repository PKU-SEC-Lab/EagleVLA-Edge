// SPDX-License-Identifier: MIT
#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <limits>

struct ggml_flashrt_patch_im2col_plan {
    const ggml_tensor * image;
    const ggml_tensor * kernel;
    ggml_tensor * output;
    int batch;
};

struct ggml_flashrt_patch_im2col_extent {
    uintptr_t begin;
    uintptr_t end;
};

// Validate dimensions before computing any extent. Even the metadata-only
// convolution weight must have the expected GGML representation; its data,
// allocation and usage are deliberately not consulted because this op reads
// only the image. The product checks also avoid trusting ggml_nbytes() on
// metadata which has not yet been validated.
static inline bool ggml_flashrt_patch_im2col_layout(
        const ggml_tensor * t, ggml_type type, const int64_t (&shape)[4],
        size_t & bytes) {
    if (!t || t->type != type ||
        (type != GGML_TYPE_F32 && type != GGML_TYPE_F16)) {
        return false;
    }
    size_t stride = type == GGML_TYPE_F32 ? sizeof(float) : sizeof(uint16_t);
    const size_t maximum = std::numeric_limits<size_t>::max();
    for (int i = 0; i < 4; ++i) {
        if (shape[i] <= 0 || t->ne[i] != shape[i] || t->nb[i] != stride ||
            uint64_t(shape[i]) > maximum / stride) {
            return false;
        }
        stride *= size_t(shape[i]);
    }
    bytes = stride;
    return true;
}

static inline bool ggml_flashrt_patch_im2col_address(
        const ggml_tensor * t, size_t bytes,
        ggml_flashrt_patch_im2col_extent & extent) {
    if (!t || !t->data) {
        return false;
    }
    const size_t alignment = t->type == GGML_TYPE_F32 ? sizeof(float) : sizeof(uint16_t);
    const uintptr_t begin = reinterpret_cast<uintptr_t>(t->data);
    if (begin % alignment != 0 ||
        bytes > std::numeric_limits<uintptr_t>::max() - begin) {
        return false;
    }
    // A contiguous GGML view is valid: the kernel starts at its resolved data
    // pointer. Reject inconsistent view metadata, without requiring the view
    // source to have this operation's particular dimensions or row layout.
    if (t->view_src) {
        const uintptr_t source = reinterpret_cast<uintptr_t>(t->view_src->data);
        if (source == 0 || t->view_offs > std::numeric_limits<uintptr_t>::max() - source ||
            source + t->view_offs != begin) {
            return false;
        }
    } else if (t->view_offs != 0) {
        return false;
    }
    extent = {begin, begin + bytes};
    return true;
}

// Pure metadata/address contract. The CUDA adapter separately verifies that
// both actually accessed ranges belong to this context's device allocation.
static inline bool ggml_flashrt_plan_patch_im2col(
        ggml_tensor * dst, ggml_flashrt_patch_im2col_plan & plan,
        const char ** rejection = nullptr) {
    if (rejection) { *rejection = "operator_or_sources"; }
    if (!dst || dst->op != GGML_OP_IM2COL || !dst->src[0] || !dst->src[1]) {
        return false;
    }
    for (int i = 2; i < GGML_MAX_SRC; ++i) {
        if (dst->src[i]) { return false; }
    }
    if (rejection) { *rejection = "stride_padding_dilation_or_2d"; }
    const int32_t expected[] = {14, 14, 0, 0, 1, 1, 1};
    for (int i = 0; i < 7; ++i) {
        if (dst->op_params[i] != expected[i]) { return false; }
    }
    const ggml_tensor * kernel = dst->src[0];
    const ggml_tensor * image = dst->src[1];
    const int64_t batch = image->ne[3];
    if (rejection) { *rejection = "batch_or_dtype"; }
    if ((batch != 1 && batch != 2) || image->type != GGML_TYPE_F32 ||
        (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) ||
        (kernel->type != GGML_TYPE_F32 && kernel->type != GGML_TYPE_F16)) {
        return false;
    }
    const int64_t image_shape[] = {224, 224, 3, batch};
    const int64_t kernel_shape[] = {14, 14, 3, 1152};
    const int64_t output_shape[] = {588, 16, 16, batch};
    size_t image_bytes = 0;
    size_t kernel_bytes = 0;
    size_t output_bytes = 0;
    if (rejection) { *rejection = "shape_or_layout"; }
    if (!ggml_flashrt_patch_im2col_layout(image, GGML_TYPE_F32, image_shape, image_bytes) ||
        !ggml_flashrt_patch_im2col_layout(kernel, kernel->type, kernel_shape, kernel_bytes) ||
        !ggml_flashrt_patch_im2col_layout(dst, dst->type, output_shape, output_bytes)) {
        return false;
    }
    ggml_flashrt_patch_im2col_extent image_extent{}, output_extent{};
    if (rejection) { *rejection = "alignment_address_or_view"; }
    if (!ggml_flashrt_patch_im2col_address(image, image_bytes, image_extent) ||
        !ggml_flashrt_patch_im2col_address(dst, output_bytes, output_extent)) {
        return false;
    }
    if (rejection) { *rejection = "image_output_overlap"; }
    if (image_extent.begin < output_extent.end && output_extent.begin < image_extent.end) {
        return false;
    }
    plan = {image, kernel, dst, int(batch)};
    if (rejection) { *rejection = nullptr; }
    return true;
}
