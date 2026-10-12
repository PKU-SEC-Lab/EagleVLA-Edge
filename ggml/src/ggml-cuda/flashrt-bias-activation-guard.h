// SPDX-License-Identifier: MIT
#pragma once

#include "flashrt-bias-residual-guard.h"

struct ggml_flashrt_bias_activation_plan {
    const ggml_tensor * x;
    const ggml_tensor * bias;
    ggml_tensor * add;
    ggml_tensor * unary;
    ggml_tensor * cont;
    ggml_tensor * output;
    int width;
    int rows;
    int skip;
    bool bias_first;
    ggml_unary_op kind;
};

static inline bool ggml_flashrt_activation_matrix(const ggml_tensor * t) {
    if (!t || !t->data || t->type != GGML_TYPE_F32 ||
        (t->ne[0] != 4304 && t->ne[0] != 1024) ||
        t->ne[1] <= 0 || t->ne[1] > 512 || t->ne[2] != 1 || t->ne[3] != 1 ||
        !ggml_is_contiguous(t) || uintptr_t(t->data) % 8 != 0) {
        return false;
    }
    return ggml_nbytes(t) <= std::numeric_limits<uintptr_t>::max() - uintptr_t(t->data);
}

// Pure host graph contract. CUDA buffer ownership and SM89 are checked at the
// dispatch boundary. A linked terminal CONT selects the three-node contract;
// if it cannot be elided safely we fall back for the entire candidate.
static inline int ggml_flashrt_plan_bias_activation(
        const ggml_cgraph * graph, int index, ggml_flashrt_bias_activation_plan & plan,
        const char ** rejection = nullptr) {
    if (rejection) {
        *rejection = nullptr;
    }
    if (!graph || index < 0 || index >= graph->n_nodes - 1) {
        return 0;
    }
    ggml_tensor * a = graph->nodes[index];
    ggml_tensor * u = graph->nodes[index + 1];
    if (!a || a->op != GGML_OP_ADD || !ggml_flashrt_bias_sources(a, 2) ||
        !ggml_flashrt_activation_matrix(a) || a->view_src ||
        !u || u->op != GGML_OP_UNARY || u->src[0] != a ||
        !ggml_flashrt_bias_sources(u, 1) || !ggml_flashrt_activation_matrix(u) ||
        !ggml_are_same_layout(a, u) || u->view_src) {
        return 0;
    }
    const ggml_unary_op kind = ggml_get_unary_op(u);
    const int64_t width = a->ne[0];
    const int64_t rows = a->ne[1];
    if (!((kind == GGML_UNARY_OP_GELU && width == 4304 &&
            (rows == 1 || rows == 50 || rows == 256 || rows == 512)) ||
          (kind == GGML_UNARY_OP_SILU && width == 1024 &&
            (rows == 1 || rows == 10 || rows == 50)))) {
        return 0;
    }

    const ggml_tensor * x = a->src[0];
    const ggml_tensor * bias = a->src[1];
    if (!ggml_flashrt_activation_matrix(x) || !ggml_flashrt_activation_matrix(bias)) {
        return 0;
    }
    bool bias_first = false;
    if (!ggml_are_same_layout(x, a) && ggml_are_same_layout(bias, a)) {
        x = a->src[1];
        bias = a->src[0];
        bias_first = true;
    }
    if (!ggml_are_same_layout(x, a) || bias->ne[0] != width || bias->ne[1] != 1) {
        return 0;
    }

    ggml_tensor * c = nullptr;
    int skip = 1;
    if (index < graph->n_nodes - 2) {
        ggml_tensor * next = graph->nodes[index + 2];
        if (next && next->op == GGML_OP_CONT && next->src[0] == u) {
            if (!ggml_flashrt_bias_sources(next, 1) ||
                !ggml_flashrt_activation_matrix(next) ||
                !ggml_are_same_layout(next, u) || next->view_src) {
                return 0;
            }
            c = next;
            skip = 2;
        }
    }
    ggml_tensor * output = c ? c : u;
    const int output_index = index + skip;
    const ggml_op direct_ops[] = {GGML_OP_ADD, GGML_OP_UNARY};
    const ggml_op cont_ops[] = {GGML_OP_ADD, GGML_OP_UNARY, GGML_OP_CONT};
    if (rejection) {
        *rejection = "graph_consumers_or_views";
    }
    if (!ggml_can_fuse_subgraph(graph, index, skip + 1, c ? cont_ops : direct_ops,
                                &output_index, 1) ||
        ggml_flashrt_bias_external_views(graph, a, c ? u : nullptr, output)) {
        return 0;
    }
    // Broadcast bias must remain read-only throughout the original and fused
    // sequence. Exact same-element matrix aliases may be allocator reuse;
    // partially overlapping matrices would introduce cross-thread hazards.
    if (rejection) {
        *rejection = "partial_or_bias_write_alias";
    }
    if (ggml_flashrt_bias_overlap(a, bias) || ggml_flashrt_bias_overlap(u, bias) ||
        ggml_flashrt_bias_overlap(output, bias) ||
        !ggml_flashrt_bias_no_partial(a, x) ||
        !ggml_flashrt_bias_no_partial(u, x) ||
        !ggml_flashrt_bias_no_partial(u, a) ||
        !ggml_flashrt_bias_no_partial(output, x) ||
        !ggml_flashrt_bias_no_partial(output, a) ||
        !ggml_flashrt_bias_no_partial(output, u)) {
        return 0;
    }
    if (rejection) {
        *rejection = "omitted_input_write_observable";
    }
    if (!ggml_flashrt_bias_exact_alias(output, x) &&
        (ggml_flashrt_bias_overlap(a, x) || (c && ggml_flashrt_bias_overlap(u, x))) &&
        ggml_flashrt_bias_x_observed_after(graph, output_index, x)) {
        return 0;
    }
    if (rejection) {
        *rejection = "omitted_intermediate_write_observable";
    }
    if (ggml_flashrt_bias_omitted_write_observed(graph, output_index, a, output) ||
        (c && ggml_flashrt_bias_omitted_write_observed(graph, output_index, u, output))) {
        return 0;
    }

    plan = {x, bias, a, u, c, output, int(width), int(rows), skip, bias_first, kind};
    if (rejection) {
        *rejection = nullptr;
    }
    return skip;
}
