// SPDX-License-Identifier: MIT
#pragma once

struct ggml_cgraph;
struct ggml_backend_cuda_context;

// FlashRT bias-residual kernel adapted to GGML F32 and explicit ADD rounding.
// Returns 1 (ADD->ADD) or 2 (ADD->CONT->ADD) after launching, 0 for fallback.
int ggml_cuda_flashrt_bias_residual(ggml_backend_cuda_context & ctx, ggml_cgraph * graph, int index);
