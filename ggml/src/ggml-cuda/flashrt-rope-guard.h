// SPDX-License-Identifier: MIT
#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

static constexpr int GGML_FLASHRT_ROPE_HEAD_DIM = 256;
static constexpr int GGML_FLASHRT_ROPE_LUT_POSITIONS = 4096;

struct ggml_flashrt_rope_plan {
    const ggml_tensor * src;
    const ggml_tensor * positions;
    ggml_tensor * dst;
    int heads;
    int rows;
    size_t tensor_bytes;
    size_t position_bytes;
};

struct ggml_flashrt_rope_extent {
    uintptr_t begin;
    uintptr_t end;
};

static inline bool ggml_flashrt_rope_layout(
        const ggml_tensor * t, ggml_type type, const int64_t (&shape)[4], size_t & bytes) {
    if (!t || t->type != type || (type != GGML_TYPE_F32 && type != GGML_TYPE_I32)) {
        return false;
    }
    size_t stride = sizeof(uint32_t);
    for (int i = 0; i < 4; ++i) {
        if (shape[i] <= 0 || t->ne[i] != shape[i] || t->nb[i] != stride ||
            uint64_t(shape[i]) > std::numeric_limits<size_t>::max() / stride) {
            return false;
        }
        stride *= size_t(shape[i]);
    }
    bytes = stride;
    return true;
}

static inline bool ggml_flashrt_rope_address(
        const ggml_tensor * t, size_t bytes, ggml_flashrt_rope_extent & extent) {
    if (!t || !t->data) { return false; }
    const uintptr_t begin = reinterpret_cast<uintptr_t>(t->data);
    if (begin % sizeof(uint32_t) != 0 || bytes > std::numeric_limits<uintptr_t>::max() - begin) {
        return false;
    }
    if (t->view_src) {
        const uintptr_t source = reinterpret_cast<uintptr_t>(t->view_src->data);
        if (!source || t->view_offs > std::numeric_limits<uintptr_t>::max() - source ||
            source + t->view_offs != begin) {
            return false;
        }
    } else if (t->view_offs) {
        return false;
    }
    extent = {begin, begin + bytes};
    return true;
}

static inline bool ggml_flashrt_rope_overlap(
        const ggml_flashrt_rope_extent & a, const ggml_flashrt_rope_extent & b) {
    return a.begin < b.end && b.begin < a.end;
}

// Pure metadata/address guard. Device ownership and allocation extents are
// checked separately immediately before the launch. Position VALUES are not
// inspected on the host; every int32 value has a defined device-side path.
static inline bool ggml_flashrt_plan_rope(
        ggml_tensor * dst, ggml_flashrt_rope_plan & plan, const char ** rejection = nullptr) {
    if (rejection) { *rejection = "operator_or_sources"; }
    if (!dst || dst->op != GGML_OP_ROPE || !dst->src[0] || !dst->src[1]) { return false; }
    for (int i = 2; i < GGML_MAX_SRC; ++i) {
        if (dst->src[i]) { return false; }
    }
    if (rejection) { *rejection = "rope_parameters"; }
    if (dst->op_params[1] != GGML_FLASHRT_ROPE_HEAD_DIM ||
        dst->op_params[2] != GGML_ROPE_TYPE_NEOX || dst->op_params[4] < 0) {
        return false;
    }
    // n_past (0), n_ctx (3), and the MRoPE sections (11..14) are ignored by
    // GGML's ordinary NEOX forward op. beta values are irrelevant with ext=0,
    // but this bounded adapter accepts only the pi0 defaults.
    const float expected[] = {10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f};
    for (int i = 0; i < 6; ++i) {
        float parameter = 0.0f;
        std::memcpy(&parameter, &dst->op_params[i + 5], sizeof(parameter));
        if (parameter != expected[i]) { return false; }
    }
    const ggml_tensor * src = dst->src[0];
    const ggml_tensor * positions = dst->src[1];
    const int64_t heads = src->ne[1];
    const int64_t rows = src->ne[2];
    if (rejection) { *rejection = "head_or_sequence_shape"; }
    if ((heads != 1 && heads != 8) || (rows != 11 && rows != 264 && rows != 268)) {
        return false;
    }
    const int64_t tensor_shape[] = {GGML_FLASHRT_ROPE_HEAD_DIM, heads, rows, 1};
    const int64_t position_shape[] = {rows, 1, 1, 1};
    size_t tensor_bytes = 0, output_bytes = 0, position_bytes = 0;
    if (rejection) { *rejection = "dtype_shape_or_layout"; }
    if (!ggml_flashrt_rope_layout(src, GGML_TYPE_F32, tensor_shape, tensor_bytes) ||
        !ggml_flashrt_rope_layout(dst, GGML_TYPE_F32, tensor_shape, output_bytes) ||
        !ggml_flashrt_rope_layout(positions, GGML_TYPE_I32, position_shape, position_bytes)) {
        return false;
    }
    ggml_flashrt_rope_extent input{}, output{}, pos{};
    if (rejection) { *rejection = "alignment_address_or_view"; }
    if (!ggml_flashrt_rope_address(src, tensor_bytes, input) ||
        !ggml_flashrt_rope_address(dst, output_bytes, output) ||
        !ggml_flashrt_rope_address(positions, position_bytes, pos)) {
        return false;
    }
    if (rejection) { *rejection = "accessed_ranges_overlap"; }
    if (ggml_flashrt_rope_overlap(input, output) || ggml_flashrt_rope_overlap(input, pos) ||
        ggml_flashrt_rope_overlap(output, pos)) {
        return false;
    }
    plan = {src, positions, dst, int(heads), int(rows), tensor_bytes, position_bytes};
    if (rejection) { *rejection = nullptr; }
    return true;
}
