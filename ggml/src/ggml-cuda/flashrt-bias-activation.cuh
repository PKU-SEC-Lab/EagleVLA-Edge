// SPDX-License-Identifier: MIT
#pragma once

struct ggml_backend_cuda_context;
struct ggml_cgraph;

int ggml_cuda_flashrt_bias_activation(ggml_backend_cuda_context & ctx, ggml_cgraph * graph, int index);
