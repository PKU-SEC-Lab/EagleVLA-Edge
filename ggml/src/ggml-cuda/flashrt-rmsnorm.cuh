#pragma once

#ifdef GGML_CUDA_FLASHRT
#include "common.cuh"

// Opt-in Gemma RMSNorm diamond; true only after a real fused launch.
bool ggml_cuda_flashrt_rmsnorm(ggml_backend_cuda_context & ctx, ggml_cgraph * graph, int node_idx);
#endif
