# FlashRT kernel attribution

Upstream: https://github.com/flashrt-project/FlashRT

Pinned revision: `e881e1424cf4fadee7dff121d01cc4e159e550e7`.

`ggml/src/ggml-cuda/flashrt-layernorm.cu` adapts `layer_norm_kernel`
from `csrc/kernels/norm.cu` and reduction routines from
`csrc/kernels/common.cuh`. These portions are under Apache-2.0;
the complete upstream license is retained in this directory. No upstream
NOTICE file was present in this revision's tracked files.

Changes: FP16/BF16 packed IO becomes FP32/float2 for ggml; the 256-thread,
two-pass normalization and affine fusion remain. A barrier protects repeated
shared-memory reductions. The epilogue rounds separate FP32 operations.
The host adapter uses ggml tensors and its existing CUDA stream, validates
layout, alignment, consumers and aliases, and opts in only on SM89.
No PyTorch, pybind11, CUTLASS or new allocator is introduced.

This is an adapted source port, not a binary-compatible FlashRT integration
or a port of FlashRT's complete inference pipeline.
