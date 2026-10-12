// SPDX-License-Identifier: MIT
#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <limits>

struct ggml_flashrt_prefill_plan {
    const ggml_tensor * weight;
    const ggml_tensor * input;
    ggml_tensor * output;
    int k;
    int n;
    int m;
};

struct ggml_flashrt_prefill_extent {
    uintptr_t begin;
    uintptr_t end;
};

static inline bool ggml_flashrt_prefill_shape_supported(int64_t k, int64_t n, int64_t m) {
    const bool vision = m == 256 && (
        (k == 1152 && n == 1152) || (k == 1152 && n == 4304) ||
        (k == 4304 && n == 1152) || (k == 1152 && n == 2048));
    const bool encoder = (m == 264 || m == 268) && (
        (k == 2048 && n == 2048) || (k == 2048 && n == 256) ||
        (k == 2048 && n == 16384) || (k == 16384 && n == 2048));
    return vision || encoder;
}

// Validate every integer/address calculation without ggml_nbytes(), which is
// only valid once the metadata has itself been proved internally consistent.
static inline bool ggml_flashrt_prefill_matrix(
        const ggml_tensor * t, ggml_type type, int64_t width, int64_t rows,
        ggml_flashrt_prefill_extent & extent) {
    if (!t || !t->data || t->type != type ||
        (type != GGML_TYPE_F16 && type != GGML_TYPE_F32) ||
        width <= 0 || rows <= 0 || t->ne[0] != width || t->ne[1] != rows ||
        t->ne[2] != 1 || t->ne[3] != 1) {
        return false;
    }
    const size_t element_bytes = type == GGML_TYPE_F16 ? sizeof(uint16_t) : sizeof(float);
    const size_t maximum = std::numeric_limits<size_t>::max();
    if (uint64_t(width) > maximum / element_bytes) {
        return false;
    }
    const size_t row_bytes = size_t(width) * element_bytes;
    if (uint64_t(rows) > maximum / row_bytes) {
        return false;
    }
    const size_t bytes = row_bytes * size_t(rows);
    const uintptr_t begin = reinterpret_cast<uintptr_t>(t->data);
    if (t->nb[0] != element_bytes || t->nb[1] != row_bytes ||
        t->nb[2] != bytes || t->nb[3] != bytes || begin % 16 != 0 ||
        bytes > std::numeric_limits<uintptr_t>::max() - begin) {
        return false;
    }
    extent = {begin, begin + bytes};
    return true;
}

static inline bool ggml_flashrt_prefill_intersects(
        const ggml_flashrt_prefill_extent & a, const ggml_flashrt_prefill_extent & b) {
    return a.begin < b.end && b.begin < a.end;
}

// Pure metadata/address guard. CUDA device-buffer ownership and immutable
// weight usage are checked separately, before any enqueue in the adapter.
static inline bool ggml_flashrt_plan_prefill_gemm(
        ggml_tensor * dst, ggml_flashrt_prefill_plan & plan,
        const char ** rejection = nullptr) {
    if (rejection) { *rejection = "operator_or_sources"; }
    if (!dst || dst->op != GGML_OP_MUL_MAT || !dst->src[0] || !dst->src[1]) {
        return false;
    }
    for (int i = 2; i < GGML_MAX_SRC; ++i) {
        if (dst->src[i]) { return false; }
    }
    if (rejection) { *rejection = "precision_or_hint"; }
    if (dst->op_params[0] != GGML_PREC_DEFAULT || dst->op_params[1] != GGML_HINT_NONE) {
        return false;
    }
    const ggml_tensor * w = dst->src[0];
    const ggml_tensor * x = dst->src[1];
    const int64_t k = w->ne[0];
    const int64_t n = w->ne[1];
    const int64_t m = x->ne[1];
    if (rejection) { *rejection = "shape_not_supported"; }
    if (!ggml_flashrt_prefill_shape_supported(k, n, m)) {
        return false;
    }
    if (rejection) { *rejection = "weight_view_or_dtype"; }
    if (w->view_src || w->view_offs != 0 ||
        (x->type != GGML_TYPE_F16 && x->type != GGML_TYPE_F32)) {
        return false;
    }
    ggml_flashrt_prefill_extent we{}, xe{}, de{};
    if (rejection) { *rejection = "layout_alignment_or_address_overflow"; }
    if (!ggml_flashrt_prefill_matrix(w, GGML_TYPE_F16, k, n, we) ||
        !ggml_flashrt_prefill_matrix(x, x->type, k, m, xe) ||
        !ggml_flashrt_prefill_matrix(dst, GGML_TYPE_F32, n, m, de)) {
        return false;
    }
    if (rejection) { *rejection = "tensor_overlap"; }
    if (ggml_flashrt_prefill_intersects(we, xe) ||
        ggml_flashrt_prefill_intersects(de, we) || ggml_flashrt_prefill_intersects(de, xe)) {
        return false;
    }
    plan = {w, x, dst, int(k), int(n), int(m)};
    if (rejection) { *rejection = nullptr; }
    return true;
}
