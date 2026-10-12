// SPDX-License-Identifier: MIT
#include "ggml-cuda/flashrt-gemm-guard.h"

#include <array>
#include <cstdio>
#include <stdexcept>

namespace {

int checks = 0;

void check(bool actual, bool expected, const char * label) {
    ++checks;
    if (actual != expected) {
        throw std::runtime_error(label);
    }
}

struct fixture {
    ggml_context * ctx = nullptr;
    ggml_tensor * w = nullptr;
    ggml_tensor * x = nullptr;
    ggml_tensor * out = nullptr;

    explicit fixture(int k = 1024, int n = 2048, int m = 11) {
        ctx = ggml_init({16 * 1024 * 1024, nullptr, false});
        if (!ctx) { throw std::runtime_error("context allocation"); }
        w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, k, n);
        x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
        out = ggml_mul_mat(ctx, w, x);
    }

    ~fixture() { ggml_free(ctx); }
    fixture(const fixture &) = delete;
    fixture & operator=(const fixture &) = delete;

    bool accepts() {
        ggml_flashrt_gemm_plan plan;
        return ggml_flashrt_plan_gemm(out, plan);
    }
};

} // namespace

int main() {
    try {
        const std::array<std::array<int, 2>, 5> shapes = {{{1024, 2048}, {1024, 256},
            {2048, 1024}, {1024, 4096}, {4096, 1024}}};
        for (const auto & shape : shapes) {
            fixture f(shape[0], shape[1]);
            ggml_flashrt_gemm_plan plan;
            check(ggml_flashrt_plan_gemm(f.out, plan), true, "supported contiguous tensor contract");
            check(plan.k == shape[0] && plan.n == shape[1] && plan.m == 11 &&
                  plan.weight == f.w && plan.input == f.x && plan.output == f.out,
                  true, "GGML ne to GEMM K/N/M mapping and source identity");
        }
        for (int rows : {1, 10, 12, 50, 256}) {
            fixture f(1024, 2048, rows);
            check(f.accepts(), false, "unsupported M must keep GGML dispatch");
        }
        { fixture f(512, 2048); check(f.accepts(), false, "unsupported K"); }
        { fixture f(1024, 1024); check(f.accepts(), false, "unsupported N"); }
        { fixture f; f.out->op = GGML_OP_ADD; check(f.accepts(), false, "non-MUL_MAT operation"); }
        { fixture f; ggml_mul_mat_set_prec(f.out, GGML_PREC_F32);
          check(f.accepts(), false, "explicit higher precision must not be overridden"); }
        { fixture f; ggml_mul_mat_set_hint(f.out, GGML_HINT_SRC0_IS_HADAMARD);
          check(f.accepts(), false, "special operation hint must not be overridden"); }
        { fixture f; f.w->type = GGML_TYPE_F32; check(f.accepts(), false, "weight dtype"); }
        { fixture f; f.x->type = GGML_TYPE_F16; check(f.accepts(), false, "input dtype"); }
        { fixture f; f.out->type = GGML_TYPE_F16; check(f.accepts(), false, "output dtype"); }
        { fixture f; f.out->src[0] = nullptr; check(f.accepts(), false, "missing weight"); }
        { fixture f; f.out->src[1] = nullptr; check(f.accepts(), false, "missing input"); }
        for (int which : {0, 1, 2}) {
            fixture f;
            ggml_tensor * t = which == 0 ? f.w : which == 1 ? f.x : f.out;
            t->data = nullptr;
            check(f.accepts(), false, "unallocated tensor");
        }
        for (int which : {0, 1, 2}) {
            fixture f;
            ggml_tensor * t = which == 0 ? f.w : which == 1 ? f.x : f.out;
            t->data = static_cast<char *>(t->data) + 4;
            check(f.accepts(), false, "less than 16 byte alignment");
        }
        for (int which : {0, 1, 2}) {
            fixture f;
            ggml_tensor * t = which == 0 ? f.w : which == 1 ? f.x : f.out;
            t->nb[1] += 16;
            check(f.accepts(), false, "noncontiguous leading dimension");
        }
        for (int which : {0, 1, 2}) {
            fixture f;
            ggml_tensor * t = which == 0 ? f.w : which == 1 ? f.x : f.out;
            t->ne[2] = 2;
            check(f.accepts(), false, "batched tensor");
        }
        { fixture f; f.w->ne[3] = 2; check(f.accepts(), false, "four dimensional batch"); }
        { fixture f; f.x->ne[0] /= 2; check(f.accepts(), false, "mismatched input K"); }
        { fixture f; f.out->ne[0] /= 2; check(f.accepts(), false, "mismatched output N"); }
        { fixture f; f.out->ne[1] = 10; check(f.accepts(), false, "mismatched output M"); }
        { fixture f; f.out->src[0] = ggml_view_tensor(f.ctx, f.w);
          check(f.accepts(), false, "weight view"); }
        { fixture f; f.out->data = f.x->data; check(f.accepts(), false, "exact input-output alias"); }
        { fixture f; f.out->data = static_cast<char *>(f.x->data) + 16;
          check(f.accepts(), false, "aligned partial input-output alias"); }
        { fixture f; f.out->data = f.w->data; check(f.accepts(), false, "exact weight-output alias"); }
        { fixture f; f.out->data = static_cast<char *>(f.w->data) + 16;
          check(f.accepts(), false, "aligned partial weight-output alias"); }
        { fixture f; f.x->data = f.w->data; check(f.accepts(), false, "weight-input alias"); }
        { fixture f; f.x->data = static_cast<char *>(f.w->data) + ggml_nbytes(f.w);
          check(f.accepts(), true, "adjacent nonoverlapping ranges remain valid"); }
        {
            ggml_flashrt_gemm_plan plan;
            check(ggml_flashrt_plan_gemm(nullptr, plan), false, "null output");
        }
        std::printf("%d GEMM tensor contract checks passed\n", checks);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "GEMM guard failed: %s\n", error.what());
        return 1;
    }
}
