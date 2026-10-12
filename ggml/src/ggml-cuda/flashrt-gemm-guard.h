// SPDX-License-Identifier: MIT
#pragma once

#include "ggml-impl.h"

#include <cstdint>

// Pure host-side tensor contract. Device ownership and immutable weight-buffer
// usage are checked separately by the CUDA adapter before any work is enqueued.
struct ggml_flashrt_gemm_plan {
    const ggml_tensor * weight;
    const ggml_tensor * input;
    ggml_tensor * output;
    int k;
    int n;
    int m;
};

static inline bool ggml_flashrt_gemm_shape_supported(int64_t k, int64_t n, int64_t m) {
    return m == 11 && (
        (k == 1024 && n == 2048) ||
        (k == 1024 && n ==  256) ||
        (k == 2048 && n == 1024) ||
        (k == 1024 && n == 4096) ||
        (k == 4096 && n == 1024));
}

static inline bool ggml_flashrt_gemm_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    const uintptr_t x = uintptr_t(a->data);
    const uintptr_t y = uintptr_t(b->data);
    // Subtraction avoids overflow at the end of a pointer range.
    return x <= y ? y - x < ggml_nbytes(a) : x - y < ggml_nbytes(b);
}

static inline bool ggml_flashrt_gemm_tensor(const ggml_tensor * t, ggml_type type) {
    return t && t->data && t->type == type && t->ne[2] == 1 && t->ne[3] == 1
        && ggml_is_contiguous(t) && uintptr_t(t->data) % 16 == 0;
}

static inline bool ggml_flashrt_plan_gemm(ggml_tensor * out, ggml_flashrt_gemm_plan & plan) {
    if (!out || out->op != GGML_OP_MUL_MAT ||
        ggml_get_op_params_i32(out, 0) != GGML_PREC_DEFAULT ||
        ggml_get_op_params_i32(out, 1) != GGML_HINT_NONE) {
        return false;
    }
    const ggml_tensor * w = out->src[0];
    const ggml_tensor * x = out->src[1];
    if (!ggml_flashrt_gemm_tensor(w, GGML_TYPE_F16) ||
        !ggml_flashrt_gemm_tensor(x, GGML_TYPE_F32) ||
        !ggml_flashrt_gemm_tensor(out, GGML_TYPE_F32) || w->view_src) {
        return false;
    }
    const int64_t k = w->ne[0];
    const int64_t n = w->ne[1];
    const int64_t m = x->ne[1];
    if (!ggml_flashrt_gemm_shape_supported(k, n, m) ||
        x->ne[0] != k || out->ne[0] != n || out->ne[1] != m ||
        ggml_flashrt_gemm_overlap(w, x) || ggml_flashrt_gemm_overlap(out, w) ||
        ggml_flashrt_gemm_overlap(out, x)) {
        return false;
    }
    plan = {w, x, out, int(k), int(n), int(m)};
    return true;
}
