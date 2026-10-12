# SM89: FlashRT RoPE rotation integration

This increment targets Jetson-PI-Edge commit `1271fa933c3f81e1cb3a71a9f09f13bdefa4461b`, the existing pi0 FP16 GGUF models, and RTX 4090. It is disabled unless both `GGML_CUDA_FLASHRT=1` and `GGML_CUDA_FLASHRT_ROPE=1` are present when the CUDA backend is created.

## What is ported

The FP16 rotation arithmetic from FlashRT's `qkv_split_rope_kvcache_fp16_kernel` is adapted to the two separate GGML Q/K tensors. Supported inputs are contiguous F32 `[256,H,M,1]`, with `H=1` or `8`, and `M=11`, `264`, or `268`. Position input is a contiguous I32 vector of length M. RoPE parameters must select full-width NEOX rotation with base10000, scale1, extension0, attention factor1, and no frequency-factor tensor. The guard checks dtype, shape, strides, view metadata, address ranges, overlaps, and CUDA device ownership. Unsupported cases keep the existing GGML path.

GGML NEOX pairs the front and back halves of a head. FlashRT's original packed representation pairs adjacent elements. The adapter changes load/store indices to account for this difference; it does not reorder or modify the loaded weights.

The native rotation uses FP16 input, cos/sin coefficients and output, with float arithmetic inside the kernel. GGML tensors remain F32 at the interface. This precision contract differs from original GGML's F32 rotation, so model output drift is recorded separately from operator-reference checks.

## Graph and lifetime contract

A coefficient table for positions 0 through4095 is created once per enabled CUDA backend, before graph capture. The table is immutable during execution and lives until the backend's graph objects are destroyed. The rotation kernel reads current positions from device memory on every execution, including graph replay. Out-of-table and negative positions use a bounds-checked device FP64 coefficient calculation, then the same FP16 rounding policy. Forward dispatch does not allocate or copy the table.

The hook is only in the independent forward RoPE entry. Existing GGML backward and fused `ROPE→VIEW→SET_ROWS` entries retain their implementation. In pi0, Q/K/V projections are separate; encoder K/V are saved through CPY, and decoder K/V use CONCAT with the prefix. Full merged QKV splitting and KV-cache writing remain separate work.

## Validation entry

Run the supplied terminal installer, or after installation:

```sh
cd "$HOME/Jetson-PI-Edge-flashrt-sm89" &&
PATH="/usr/sbin:/usr/bin:/sbin:/bin:$PATH" /usr/bin/python3 scripts/run-sm89-rope.py
```

The runner consumes the accepted `patch-im2col-0faleo4x/summary.json` result and retains its original CPU/GPU placement. It holds GGML-math norms, GEMM2, bias/residual and patch_im2col enabled; GEGLU, prefill GEMM and bias/activation stay disabled. It toggles only the new RoPE switch, using separate processes for each mode and Graph setting. A single idle GPU UUID is selected for the run. Existing processes are not stopped.

Operator fixtures cover Q and K encoder/decoder shapes, a contiguous input view, position boundaries, and unsupported-path fallbacks. Each graph is run repeatedly, then receives a position-only update, then an input-only update. Complete output arrays and metadata are saved. Model validation covers fixed, changed-text and changed-state inputs, with repeated requests and Graph on/off. Model TRACE must prove both Q/K heads at all three supported sequence lengths; a startup probe or all-fallback run cannot satisfy this gate.

Baseline actions must match the previously accepted result. Candidate actions must be finite and repeatable; initial noise must remain identical, and the same candidate must agree across Graph settings. The candidate's final-action difference from baseline is reported without inventing a task-accuracy tolerance. This runner does not measure performance or certify robotic task accuracy.

## Completion boundary

`rope_rotation_integrated` means this rotation adapter's specified checks completed on the target GPU. It does not mean all FlashRT kernels are ported, that the full QKV/RoPE/KV fusion is present, or that an end-to-end speedup or acceptable task accuracy has been established. Earlier GEGLU and prefill GEMM remain experimental because their recorded model drift has not been accepted against a task-level evaluation.
