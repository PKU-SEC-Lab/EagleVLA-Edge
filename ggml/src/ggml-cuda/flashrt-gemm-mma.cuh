#pragma once

struct ggml_tensor;
struct ggml_backend_cuda_context;

// GGML-math SM89 candidate, not an upstream FlashRT native GEMM kernel.
// Returns false without enqueueing work when the opt-in or tensor contract fails.
bool ggml_cuda_flashrt_gemm_mma(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
