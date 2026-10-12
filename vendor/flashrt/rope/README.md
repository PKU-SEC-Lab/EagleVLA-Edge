# FlashRT RoPE rotation source

Upstream: https://github.com/flashrt-project/FlashRT

Pinned commit: `e881e1424cf4fadee7dff121d01cc4e159e550e7`.

Sources:

- `csrc/kernels/rope.cu`, Q/K rotation branches of `qkv_split_rope_kvcache_fp16_kernel`.
- `flash_rt/models/pi0/pipeline_rtx.py`, `_build_rope_table`.

The adapter retains FP16 input/coefficient/output semantics, with float products and sums. It adapts separate GGML F32 Q/K tensors to the upstream rotation and writes back F32 containers holding FP16-rounded values. GGML NEOX pairs `(d,d+128)` map to upstream interleaved pairs; weights are not changed. The adapter does not implement upstream merged QKV splitting, V copying, or KV-cache writing.

The coefficient table uses FP64 host `pow`, `cos`, and `sin` before FP16 rounding, matching the upstream construction's precision policy. Different math libraries are not promised bitwise identical. A per-backend table is initialized before graph capture. Positions outside the table use device FP64 coefficient construction with the stored inverse frequencies, avoiding out-of-range reads.

Complete upstream Apache-2.0 license is included as `LICENSE`. Integration glue and pure host metadata checks are MIT-licensed where indicated.
