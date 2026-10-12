#pragma once

#include "ggml-impl.h"
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>

// Host-only recognition shared by CUDA dispatch and CPU contract tests.
// Gemma: n = RMS_NORM(x); m = n * w; y = m + n. Both uses of n must be internal.
struct ggml_flashrt_rmsnorm_plan {
    const ggml_tensor * x;
    const ggml_tensor * weight;
    const ggml_tensor * norm;
    const ggml_tensor * mul;
    ggml_tensor * out;
    int dim;
    int64_t rows;
    float eps;
};

static inline bool ggml_flashrt_f32_contiguous(const ggml_tensor * t) {
    return t && t->data && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t)
        && uintptr_t(t->data) % alignof(float) == 0;
}

static inline bool ggml_flashrt_tensor_overlaps(const ggml_tensor * a, const ggml_tensor * b) {
    const uintptr_t x = uintptr_t(a->data);
    const uintptr_t y = uintptr_t(b->data);
    return x <= y ? y - x < ggml_nbytes(a) : x - y < ggml_nbytes(b);
}

static inline bool ggml_flashrt_plan_rmsnorm(
        const ggml_cgraph * graph, int i, ggml_flashrt_rmsnorm_plan & plan) {
    if (!graph || i < 0 || i > graph->n_nodes - 3 ||
        !ggml_can_fuse_subgraph(graph, i,
            { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD }, { i + 2 })) {
        return false;
    }
    const ggml_tensor * norm = graph->nodes[i];
    const ggml_tensor * mul = graph->nodes[i + 1];
    ggml_tensor * out = graph->nodes[i + 2];
    const ggml_tensor * x = norm->src[0];
    const ggml_tensor * w = mul->src[1];
    if (mul->src[0] != norm || out->src[0] != mul || out->src[1] != norm ||
        !ggml_flashrt_f32_contiguous(x) || !ggml_flashrt_f32_contiguous(w) ||
        !ggml_flashrt_f32_contiguous(norm) || !ggml_flashrt_f32_contiguous(mul) ||
        !ggml_flashrt_f32_contiguous(out) ||
        !ggml_are_same_shape(x, norm) || !ggml_are_same_shape(x, mul) ||
        !ggml_are_same_shape(x, out)) {
        return false;
    }
    const int64_t dim = x->ne[0];
    const int64_t rows = ggml_nrows(x);
    if (dim < 1 || dim > 8192 || rows < 1 || rows > INT_MAX ||
        w->ne[0] != dim || ggml_nelements(w) != dim ||
        ggml_flashrt_tensor_overlaps(out, x) || ggml_flashrt_tensor_overlaps(out, w)) {
        return false;
    }
    float eps;
    std::memcpy(&eps, norm->op_params, sizeof(eps));
    if (!std::isfinite(eps) || eps <= 0.0f) {
        return false;
    }
    plan = { x, w, norm, mul, out, int(dim), rows, eps };
    return true;
}
