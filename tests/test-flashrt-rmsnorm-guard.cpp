#include "ggml-cuda/flashrt-rmsnorm-guard.h"

#include <cstdio>
#include <stdexcept>
#include <limits>

static int checks = 0;
static void check(bool actual, bool expected, const char * label) {
    ++checks;
    if (actual != expected) {
        throw std::runtime_error(label);
    }
}

struct fixture {
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph;
    ggml_tensor * x;
    ggml_tensor * w;
    ggml_tensor * norm;
    ggml_tensor * mul;
    ggml_tensor * out;
    int index = -1;

    fixture(int width = 1024, int rows = 11, int variant = 0) {
        ctx = ggml_init({16 * 1024 * 1024, nullptr, false});
        if (!ctx) {
            throw std::runtime_error("context allocation");
        }
        x = variant == 6 ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, rows, 2)
            : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, variant == 1 ? width * 2 : width, rows);
        if (variant == 1) {
            x = ggml_view_2d(ctx, x, width, rows, x->nb[1], 0);
        }
        w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, variant == 3 ? rows : 1);
        norm = ggml_rms_norm(ctx, x, variant == 5 ? 0.0f : 1e-6f);
        mul = ggml_mul(ctx, norm, w);
        out = ggml_add(ctx, mul, variant == 10 ? x : norm);
        if (variant == 11) {
            out = ggml_add(ctx, norm, mul);
        }
        ggml_tensor * final = out;
        if (variant == 2) {
            final = ggml_add(ctx, out, norm);
        }
        if (variant == 8) {
            final = ggml_add(ctx, out, mul);
        }
        if (variant == 7) {
            ggml_set_output(norm);
        }
        if (variant == 9) {
            ggml_set_output(mul);
        }
        graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, final);
        for (int i = 0; i < graph->n_nodes; ++i) {
            if (graph->nodes[i] == norm) {
                index = i;
            }
        }
    }
    ~fixture() { ggml_free(ctx); }
    bool accepts() {
        ggml_flashrt_rmsnorm_plan plan;
        return ggml_flashrt_plan_rmsnorm(graph, index, plan);
    }
};

int main() {
    try {
        for (int width : {1, 17, 1023, 1024, 2048, 8192}) {
            fixture f(width);
            check(f.accepts(), true, "contiguous Gemma diamond must fuse");
            check(ggml_can_fuse(f.graph, f.index,
                {GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD}), false,
                "linear fusion must reject the norm's two uses");
        }
        for (int variant : {1, 2, 3, 5, 7, 8, 9, 10, 11}) {
            fixture f(1024, 11, variant);
            check(f.accepts(), false, "external uses, output flags, layout or wrong edges");
        }
        {
            fixture f(2048, 3, 6);
            ggml_flashrt_rmsnorm_plan plan;
            check(ggml_flashrt_plan_rmsnorm(f.graph, f.index, plan), true, "3D contiguous input");
            check(plan.rows == 6 && plan.dim == 2048, true, "3D flattening");
        }
        { fixture f(8193); check(f.accepts(), false, "width guard"); }
        { fixture f; f.out->data = f.x->data; check(f.accepts(), false, "exact input alias"); }
        { fixture f; f.out->data = static_cast<char *>(f.x->data) + 4; check(f.accepts(), false, "partial input alias"); }
        { fixture f; f.out->data = f.w->data; check(f.accepts(), false, "weight alias"); }
        { fixture f; f.w->type = GGML_TYPE_F16; check(f.accepts(), false, "weight dtype"); }
        { fixture f; f.out->data = nullptr; check(f.accepts(), false, "missing allocation"); }
        { fixture f; f.x->data = static_cast<char *>(f.x->data) + 1; check(f.accepts(), false, "float alignment"); }
        { fixture f; f.mul->flags &= ~GGML_TENSOR_FLAG_COMPUTE; check(f.accepts(), false, "noncomputed intermediate"); }
        {
            fixture f;
            const float bad = std::numeric_limits<float>::quiet_NaN();
            std::memcpy(f.norm->op_params, &bad, sizeof(bad));
            check(f.accepts(), false, "nonfinite eps");
        }
        {
            fixture f;
            ggml_flashrt_rmsnorm_plan plan;
            check(ggml_flashrt_plan_rmsnorm(f.graph, -1, plan), false, "negative index");
            check(ggml_flashrt_plan_rmsnorm(f.graph, f.graph->n_nodes, plan), false, "past end");
        }
        std::printf("%d RMSNorm graph/alias contract checks passed\n", checks);
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "RMSNorm guard failed: %s\n", e.what());
        return 1;
    }
}
