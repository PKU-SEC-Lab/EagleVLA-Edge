# Upstream attribution

- Repository: https://github.com/flashrt-project/FlashRT
- Commit: `e881e1424cf4fadee7dff121d01cc4e159e550e7`
- Source: `csrc/gemm/gemm_runner.cu`, FP16_NN setup and `GemmRunner::fp16_nn`
- Interface: `csrc/gemm/gemm_runner.h`
- Caller: `flash_rt/models/pi0/pipeline_rtx.py`, non-FP8 encoder path
- License: Apache-2.0, reproduced in full in `LICENSE`

Adapted file: `ggml/src/ggml-cuda/flashrt-prefill-gemm.cu`.
Changes include GGML layout/stream integration, explicit F32/F16 boundary
conversions, SM89 bounded shape dispatch, context-owned resource lifetime,
pre-capture preparation, and a 32 MiB workspace cap. Native F16 operands,
FP32 accumulation, and F16 core output are preserved. This is an adaptation
of FlashRT's cuBLASLt execution path, not a verbatim copy of the full runner
and not a claim that CUDA/cuBLASLt itself originates in FlashRT.

The metadata guard, thin header, integration hooks, tests and runner are
new adapter/support code. Existing kernels and their provenance are unchanged.
