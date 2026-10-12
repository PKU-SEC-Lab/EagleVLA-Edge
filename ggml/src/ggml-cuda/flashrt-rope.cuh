// SPDX-License-Identifier: Apache-2.0
#pragma once

struct ggml_backend_cuda_context;
struct ggml_tensor;

#ifdef GGML_CUDA_FLASHRT
// Called only at backend creation, before any CUDA graph capture. Host/device
// allocation exhaustion leaves the optional path unavailable and GGML is used;
// other CUDA errors fail explicitly, as they do in the existing backend.
void ggml_cuda_flashrt_rope_init(ggml_backend_cuda_context & ctx);
bool ggml_cuda_flashrt_rope(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
#endif
