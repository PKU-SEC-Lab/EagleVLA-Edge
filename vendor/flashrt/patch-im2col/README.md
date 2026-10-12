# Upstream attribution

- Repository: https://github.com/flashrt-project/FlashRT
- Commit: `e881e1424cf4fadee7dff121d01cc4e159e550e7`
- Source: `csrc/kernels/patch_embed.cu`, `patch_im2col_kernel` / `patch_im2col`
- Source blob: `32fe2ba1866099c0623f548daa96caa2a399ccb6`
- License: Apache-2.0, reproduced in full in `LICENSE`

Adapted file: `ggml/src/ggml-cuda/flashrt-patch-im2col.cu`.
Changes: GGML CHW input and channel-first patch features, F32 input with
F32/F16 output, fixed supported batch range, shape/layout/address/device
guards, current-stream dispatch and independent opt-in logging. Original
image preprocessing, uint8/LUT gather, patch projection and bias/position
fusion are outside this increment. The guard, thin header, integration hooks,
tests and runner are new support code.
