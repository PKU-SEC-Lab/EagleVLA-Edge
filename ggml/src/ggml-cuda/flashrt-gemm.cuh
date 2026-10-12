// SPDX-License-Identifier: MIT
#pragma once

#include "common.cuh"

// Opaque per-context, per-stream ownership. common.cuh forward-declares this
// type and stores shared_ptr instances in flashrt_gemm_states[]. Destruction
// must happen inside ggml_backend_cuda_context's existing CUDA capture lock,
// before its streams are destroyed.
struct ggml_cuda_flashrt_gemm_state;

// Experimental, explicitly gated by both GGML_CUDA_FLASHRT=1 and
// GGML_CUDA_FLASHRT_GEMM=1. Converts F32 activations to F16; multiplies F16
// weights and activations with FP32 accumulation and writes F32 outputs.
// This is not an assertion of bitwise agreement with GGML's half2 reduction.
// false means no destination write was enqueued, so ordinary GGML may run.
// CUDA/cuBLAS execution failures are fatal rather than silently falling back.
bool ggml_cuda_flashrt_gemm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
