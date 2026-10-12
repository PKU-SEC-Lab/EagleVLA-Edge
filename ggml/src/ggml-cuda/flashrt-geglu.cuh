// SPDX-License-Identifier: MIT
#pragma once

struct ggml_backend_cuda_context;
struct ggml_tensor;

// Returns true only after the FlashRT GEGLU kernel was enqueued on ctx.stream().
// Returning false leaves dst unchanged and requests the original GGML path.
bool ggml_cuda_flashrt_geglu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
