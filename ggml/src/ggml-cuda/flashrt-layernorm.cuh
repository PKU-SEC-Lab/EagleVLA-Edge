#pragma once

#ifdef GGML_CUDA_FLASHRT
#include "common.cuh"

// True only when a supported fusion was actually enqueued on ctx.stream().
bool ggml_cuda_flashrt_layernorm(ggml_backend_cuda_context & ctx, ggml_cgraph * graph, int node_idx);
#endif
