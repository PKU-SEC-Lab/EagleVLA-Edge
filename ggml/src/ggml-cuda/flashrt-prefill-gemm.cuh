// SPDX-License-Identifier: MIT
#pragma once

struct ggml_backend_cuda_context;
struct ggml_tensor;
struct ggml_cuda_flashrt_prefill_state;

// Returns false only before writing dst. A true return means the FlashRT
// FP16 GEMM and FP16-to-GGML-F32 output adapter were enqueued on ctx.stream().
bool ggml_cuda_flashrt_prefill_gemm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
