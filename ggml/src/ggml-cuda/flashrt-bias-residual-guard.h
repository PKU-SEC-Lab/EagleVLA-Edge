// SPDX-License-Identifier: MIT
#pragma once

#include "ggml-impl.h"

#include <cstdint>
#include <limits>

struct ggml_flashrt_bias_residual_plan {
    const ggml_tensor * x;
    const ggml_tensor * bias;
    const ggml_tensor * residual;
    ggml_tensor * add;
    ggml_tensor * cont;
    ggml_tensor * output;
    int width;
    int rows;
    int skip;
    bool bias_first;
    bool residual_first;
};

static inline bool ggml_flashrt_bias_matrix(const ggml_tensor * t) {
    if (!t || !t->data || t->type != GGML_TYPE_F32 ||
        t->ne[0] != 1152 || t->ne[1] <= 0 || t->ne[1] > 512 ||
        t->ne[2] != 1 || t->ne[3] != 1 || !ggml_is_contiguous(t) ||
        uintptr_t(t->data) % 8 != 0) {
        return false;
    }
    const uintptr_t address = uintptr_t(t->data);
    return ggml_nbytes(t) <= std::numeric_limits<uintptr_t>::max() - address;
}

static inline bool ggml_flashrt_bias_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    const uintptr_t x = uintptr_t(a->data);
    const uintptr_t y = uintptr_t(b->data);
    return x <= y ? y - x < ggml_nbytes(a) : x - y < ggml_nbytes(b);
}

static inline bool ggml_flashrt_bias_exact_alias(const ggml_tensor * a, const ggml_tensor * b) {
    return a->data == b->data && ggml_are_same_layout(a, b);
}

static inline bool ggml_flashrt_bias_no_partial(const ggml_tensor * a, const ggml_tensor * b) {
    return !ggml_flashrt_bias_overlap(a, b) || ggml_flashrt_bias_exact_alias(a, b);
}

static inline bool ggml_flashrt_bias_sources(const ggml_tensor * t, int count) {
    for (int i = count; i < GGML_MAX_SRC; ++i) {
        if (t->src[i]) {
            return false;
        }
    }
    return true;
}

static inline bool ggml_flashrt_bias_view_of(const ggml_tensor * t, const ggml_tensor * source) {
    // ggml_new_tensor_impl collapses ordinary view chains. Bound traversal as
    // a defensive fallback for manually constructed, invalid cyclic tensors.
    for (int depth = 0; t && depth < 64; ++depth, t = t->view_src) {
        if (t == source) {
            return true;
        }
        if (!t->view_src) {
            return false;
        }
    }
    return t != nullptr;
}

static inline bool ggml_flashrt_bias_external_views(
        const ggml_cgraph * graph, const ggml_tensor * a, const ggml_tensor * c,
        const ggml_tensor * output) {
    // Raw ggml_view_tensor() sets view_src without adding src[0]; therefore the
    // graph's src[] use counts alone do not exclude every observable view.
    for (int group = 0; group < 2; ++group) {
        const int count = group == 0 ? graph->n_nodes : graph->n_leafs;
        ggml_tensor * const * tensors = group == 0 ? graph->nodes : graph->leafs;
        for (int i = 0; i < count; ++i) {
            const ggml_tensor * t = tensors[i];
            if (t == a || t == c || t == output || !t || !t->view_src) {
                continue;
            }
            if (ggml_flashrt_bias_view_of(t->view_src, a) ||
                (c && ggml_flashrt_bias_view_of(t->view_src, c))) {
                return true;
            }
        }
    }
    return false;
}

static inline const ggml_tensor * ggml_flashrt_bias_view_root(const ggml_tensor * t) {
    for (int depth = 0; t && t->view_src && depth < 64; ++depth) {
        t = t->view_src;
    }
    return t;
}

static inline bool ggml_flashrt_bias_observable(const ggml_tensor * t) {
    for (int depth = 0; t && depth < 64; ++depth, t = t->view_src) {
        if (t->flags & (GGML_TENSOR_FLAG_INPUT | GGML_TENSOR_FLAG_OUTPUT)) {
            return true;
        }
    }
    return t != nullptr;
}

static inline bool ggml_flashrt_bias_omitted_write_observed(
        const ggml_cgraph * graph, int last, const ggml_tensor * intermediate, const ggml_tensor * output) {
    if (!intermediate || ggml_flashrt_bias_exact_alias(intermediate, output)) {
        return false;
    }
    for (int group = 0; group < 2; ++group) {
        const int count = group == 0 ? graph->n_nodes : graph->n_leafs;
        ggml_tensor * const * tensors = group == 0 ? graph->nodes : graph->leafs;
        for (int i = 0; i < count; ++i) {
            const ggml_tensor * t = tensors[i];
            // A future computed tensor may legitimately reuse this now-dead
            // intermediate's storage. Its later write does not observe ours.
            if (t && group == 0 && i > last && t->op != GGML_OP_NONE &&
                !t->view_src && !(t->flags & GGML_TENSOR_FLAG_INPUT)) {
                continue;
            }
            if (t && t->data && ggml_flashrt_bias_observable(t) &&
                ggml_flashrt_bias_overlap(t, intermediate)) {
                return true;
            }
        }
    }
    return false;
}

static inline bool ggml_flashrt_bias_x_observed_after(
        const ggml_cgraph * graph, int last, const ggml_tensor * x) {
    if (ggml_flashrt_bias_observable(x)) {
        return true;
    }
    const ggml_tensor * root = ggml_flashrt_bias_view_root(x);
    for (int i = last + 1; i < graph->n_nodes; ++i) {
        const ggml_tensor * t = graph->nodes[i];
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            const ggml_tensor * source = t->src[s];
            if (source && (ggml_flashrt_bias_view_of(source, x) ||
                (source->data && ggml_flashrt_bias_view_root(source) == root &&
                 ggml_flashrt_bias_overlap(source, x)))) {
                return true;
            }
        }
        if (t->view_src && ggml_flashrt_bias_view_of(t->view_src, x)) {
            return true;
        }
    }
    for (int i = 0; i < graph->n_leafs; ++i) {
        const ggml_tensor * t = graph->leafs[i];
        if (t != x && t->view_src && ggml_flashrt_bias_view_of(t->view_src, x)) {
            return true;
        }
    }
    return false;
}

// Pure host contract, suitable for tests without CUDA. Buffer ownership and
// architecture are validated by the CUDA entry point before any launch.
// Returns the number of subsequent graph nodes to skip, or zero for fallback.
static inline int ggml_flashrt_plan_bias_residual(
        const ggml_cgraph * graph, int index, ggml_flashrt_bias_residual_plan & plan,
        const char ** rejection = nullptr) {
    if (rejection) {
        *rejection = nullptr;
    }
    if (!graph || index < 0 || index >= graph->n_nodes - 1) {
        return 0;
    }
    ggml_tensor * a = graph->nodes[index];
    if (!a || a->op != GGML_OP_ADD || !ggml_flashrt_bias_sources(a, 2) ||
        !ggml_flashrt_bias_matrix(a) || a->view_src) {
        return 0;
    }
    const int64_t rows = a->ne[1];
    if (rows != 1 && rows != 50 && rows != 256 && rows != 512) {
        return 0;
    }
    ggml_tensor * c = nullptr;
    int skip = 1;
    ggml_tensor * output = graph->nodes[index + 1];
    if (output && output->op == GGML_OP_CONT) {
        c = output;
        skip = 2;
        if (index >= graph->n_nodes - 2 || c->src[0] != a ||
            !ggml_flashrt_bias_sources(c, 1) || !ggml_flashrt_bias_matrix(c) ||
            !ggml_are_same_layout(c, a) || c->view_src) {
            return 0;
        }
        output = graph->nodes[index + 2];
    }
    if (!output || output->op != GGML_OP_ADD ||
        !ggml_flashrt_bias_sources(output, 2) || !ggml_flashrt_bias_matrix(output) ||
        !ggml_are_same_layout(output, a) || output->view_src) {
        return 0;
    }

    const ggml_tensor * x = a->src[0];
    const ggml_tensor * bias = a->src[1];
    if (!ggml_flashrt_bias_matrix(x) || !ggml_flashrt_bias_matrix(bias)) {
        return 0;
    }
    bool bias_first = false;
    if (!ggml_are_same_layout(x, a) && ggml_are_same_layout(bias, a)) {
        x = a->src[1];
        bias = a->src[0];
        bias_first = true;
    }
    if (!ggml_are_same_layout(x, a) || bias->ne[1] != 1) {
        return 0;
    }
    const ggml_tensor * previous = c ? c : a;
    const bool residual_first = output->src[1] == previous;
    if (output->src[0] != previous && !residual_first) {
        return 0;
    }
    const ggml_tensor * residual = output->src[residual_first ? 0 : 1];
    if (residual == previous || !ggml_flashrt_bias_matrix(residual) ||
        !ggml_are_same_layout(residual, a)) {
        return 0;
    }

    const ggml_op direct_ops[] = {GGML_OP_ADD, GGML_OP_ADD};
    const ggml_op cont_ops[] = {GGML_OP_ADD, GGML_OP_CONT, GGML_OP_ADD};
    const int output_index = index + skip;
    if (rejection) {
        *rejection = "graph_consumers_or_views";
    }
    if (!ggml_can_fuse_subgraph(graph, index, skip + 1, c ? cont_ops : direct_ops,
                                &output_index, 1) ||
        ggml_flashrt_bias_external_views(graph, a, c, output)) {
        return 0;
    }

    // Preserve original program reads: its intermediate writes must not change
    // the residual or the broadcast bias before the final ADD reads them.
    if (rejection) {
        *rejection = "intermediate_overwrites_residual_or_bias";
    }
    if (ggml_flashrt_bias_overlap(a, residual) || ggml_flashrt_bias_overlap(a, bias) ||
        (c && (ggml_flashrt_bias_overlap(c, residual) || ggml_flashrt_bias_overlap(c, bias)))) {
        return 0;
    }
    if (rejection) {
        *rejection = "partial_or_output_bias_alias";
    }
    if (ggml_flashrt_bias_overlap(output, bias) || !ggml_flashrt_bias_no_partial(a, x) ||
        !ggml_flashrt_bias_no_partial(output, x) ||
        !ggml_flashrt_bias_no_partial(output, residual) ||
        !ggml_flashrt_bias_no_partial(output, a) ||
        !ggml_flashrt_bias_no_partial(x, residual) ||
        (c && (!ggml_flashrt_bias_no_partial(c, x) ||
               !ggml_flashrt_bias_no_partial(c, a) ||
               !ggml_flashrt_bias_no_partial(output, c)))) {
        return 0;
    }
    // The allocator may reuse X for ADD/CONT without setting view_src. Omitting
    // those writes is safe only when later graph operations cannot observe X.
    if (rejection) {
        *rejection = "omitted_input_write_observable";
    }
    if (!ggml_flashrt_bias_exact_alias(output, x) &&
        (ggml_flashrt_bias_overlap(a, x) || (c && ggml_flashrt_bias_overlap(c, x))) &&
        ggml_flashrt_bias_x_observed_after(graph, output_index, x)) {
        return 0;
    }
    if (rejection) {
        *rejection = "omitted_intermediate_write_observable";
    }
    if (ggml_flashrt_bias_omitted_write_observed(graph, output_index, a, output) ||
        ggml_flashrt_bias_omitted_write_observed(graph, output_index, c, output)) {
        return 0;
    }

    plan = {x, bias, residual, a, c, output, 1152, int(rows), skip, bias_first, residual_first};
    if (rejection) {
        *rejection = nullptr;
    }
    return skip;
}
