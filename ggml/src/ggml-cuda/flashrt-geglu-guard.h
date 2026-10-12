// SPDX-License-Identifier: MIT
#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <limits>

struct ggml_flashrt_geglu_plan {
    const ggml_tensor * gate;
    const ggml_tensor * up;
    ggml_tensor * output;
    int width;
    int rows;
    int elements;
    // Strides and offsets are in elements of the common input/output dtype.
    size_t gate_row_stride;
    size_t up_row_stride;
    size_t gate_offset;
    size_t up_offset;
    bool merged;
    bool swapped;
};

struct ggml_flashrt_geglu_extent {
    uintptr_t begin;
    uintptr_t end;
};

// A single plane with contiguous columns and optionally padded rows. Higher
// dimensions are singleton; their unused strides do not participate in loads.
// Bound every address calculation before constructing an exclusive end.
static inline bool ggml_flashrt_geglu_matrix(
        const ggml_tensor * t, enum ggml_type type, int64_t width, int64_t rows,
        ggml_flashrt_geglu_extent & extent) {
    if (!t || !t->data || t->type != type ||
        t->ne[0] != width || t->ne[1] != rows ||
        t->ne[2] != 1 || t->ne[3] != 1 || width <= 0 || rows <= 0) {
        return false;
    }
    const size_t element_size = type == GGML_TYPE_F32 ? sizeof(float) : sizeof(uint16_t);
    const uintptr_t address = reinterpret_cast<uintptr_t>(t->data);
    const size_t maximum = std::numeric_limits<size_t>::max();
    if (uint64_t(width) > maximum / element_size || t->nb[0] != element_size ||
        address % element_size != 0 || t->nb[1] % element_size != 0) {
        return false;
    }
    const size_t row_bytes = size_t(width) * element_size;
    if (t->nb[1] < row_bytes ||
        uint64_t(rows - 1) > (maximum - row_bytes) / t->nb[1]) {
        return false;
    }
    const size_t span = size_t(rows - 1) * t->nb[1] + row_bytes;
    if (span > std::numeric_limits<uintptr_t>::max() - address) {
        return false;
    }
    extent = {address, address + span};
    return true;
}

static inline bool ggml_flashrt_geglu_intersects(
        const ggml_flashrt_geglu_extent & a, const ggml_flashrt_geglu_extent & b) {
    return a.begin < b.end && b.begin < a.end;
}

// Pure metadata/address contract; no CUDA runtime or graph rewriting. Device
// and environment checks remain at the CUDA entry point. Rejections dispatch
// the original GGML operator. Input-input aliasing is read-only and harmless;
// all output-input aliasing is conservatively rejected, including row padding.
static inline bool ggml_flashrt_plan_geglu(
        ggml_tensor * dst, ggml_flashrt_geglu_plan & plan,
        const char ** rejection = nullptr) {
    if (rejection) {
        *rejection = "operator_or_sources";
    }
    if (!dst || dst->op != GGML_OP_GLU ||
        dst->op_params[0] != GGML_GLU_OP_GEGLU || !dst->src[0]) {
        return false;
    }
    for (int i = 2; i < GGML_MAX_SRC; ++i) {
        if (dst->src[i]) {
            return false;
        }
    }
    if (rejection) {
        *rejection = "dtype_or_shape";
    }
    const enum ggml_type type = dst->type;
    const int64_t width = dst->ne[0];
    const int64_t rows = dst->ne[1];
    const int64_t maximum_elements = std::numeric_limits<int>::max() - 255;
    if ((type != GGML_TYPE_F32 && type != GGML_TYPE_F16) ||
        (width != 4096 && width != 16384) ||
        rows <= 0 || rows > maximum_elements / width) {
        return false;
    }
    // GGML only applies swapped to the two halves of a merged input; for the
    // split form src[0] is always the operand receiving GELU and src[1] is up.
    const int32_t swapped = dst->op_params[1];
    if (swapped != 0 && swapped != 1) {
        return false;
    }
    const ggml_tensor * gate = dst->src[0];
    const bool merged = dst->src[1] == nullptr;
    const ggml_tensor * up = merged ? gate : dst->src[1];
    const size_t element_size = type == GGML_TYPE_F32 ? sizeof(float) : sizeof(uint16_t);
    ggml_flashrt_geglu_extent output_extent{}, gate_extent{}, up_extent{};
    if (rejection) {
        *rejection = "layout_or_address_overflow";
    }
    if (!ggml_flashrt_geglu_matrix(dst, type, width, rows, output_extent) ||
        !ggml_flashrt_geglu_matrix(gate, type, merged ? 2 * width : width, rows, gate_extent) ||
        !ggml_flashrt_geglu_matrix(up, type, merged ? 2 * width : width, rows, up_extent)) {
        return false;
    }
    const size_t output_row_bytes = size_t(width) * element_size;
    // Standard contiguous GGML output metadata, including singleton strides.
    if (uint64_t(rows) > std::numeric_limits<size_t>::max() / output_row_bytes ||
        dst->nb[1] != output_row_bytes ||
        dst->nb[2] != output_row_bytes * size_t(rows) || dst->nb[3] != dst->nb[2]) {
        return false;
    }
    if (rejection) {
        *rejection = "output_input_overlap";
    }
    if (ggml_flashrt_geglu_intersects(output_extent, gate_extent) ||
        ggml_flashrt_geglu_intersects(output_extent, up_extent)) {
        return false;
    }

    plan = {gate, up, dst, int(width), int(rows), int(width * rows),
            gate->nb[1] / element_size, up->nb[1] / element_size,
            merged && swapped ? size_t(width) : 0,
            merged && !swapped ? size_t(width) : 0, merged, swapped != 0};
    if (rejection) {
        *rejection = nullptr;
    }
    return true;
}
