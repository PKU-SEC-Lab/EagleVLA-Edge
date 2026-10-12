// SPDX-License-Identifier: MIT
#pragma once

struct ggml_backend_cuda_context;
struct ggml_tensor;

// Returns true only after the adapted FlashRT patch gather was enqueued on
// ctx.stream(). A false result leaves dst untouched for the GGML fallback.
bool ggml_cuda_flashrt_patch_im2col(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
