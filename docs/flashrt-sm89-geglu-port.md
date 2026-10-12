# FlashRT GEGLU on SM89

This increment ports FlashRT's expf GEGLU device formula into Jetson-PI-Edge's
existing GGML_OP_GLU / GGML_GLU_OP_GEGLU CUDA dispatch. It does not change model
graphs, tensor placement, model weights, or the previous norm/GEMM kernels.

## Source and changes

- Jetson-PI-Edge baseline: `1271fa933c3f81e1cb3a71a9f09f13bdefa4461b`.
- FlashRT: `e881e1424cf4fadee7dff121d01cc4e159e550e7`.
- Upstream file: https://github.com/flashrt-project/FlashRT/blob/e881e1424cf4fadee7dff121d01cc4e159e550e7/csrc/kernels/activation.cu
- Upstream file SHA256: `723462df3069dff956aea4493fd3b43fc1ae7af748a76cd6edb4a25ef46bb53f`.
- Derived functions: `gate_geglu_kernel<T>` and `gate_geglu_merged_kernel<T>`.
- License: Apache-2.0; full upstream license is in `vendor/flashrt/geglu/LICENSE`.

The formula remains `g / (1 + expf(-1.5957691216057308f*g*(1 + 0.044715f*g*g))) * u`.
The adapter supports native FP16 inputs/output with FP32 scalar arithmetic and
an F32 dtype extension, because GGML MUL_MAT outputs are F32 in these graphs.
It adds SM89 checks, split/merged addressing, swapped merged halves, independent
input row strides, overflow/alias guards, and optional capture tracing.
No intermediate dtype conversions or temporary allocation are introduced.
Gate is GGML src[0]; up is src[1] in the split representation.

## Coverage and fallback

Supported widths are 4096 (decoder FFN) and 16384 (encoder FFN), with positive
row counts bounded by the kernel's signed integer indexing. Inputs must be 2D
F32 or F16 on the same CUDA device, with contiguous columns. Row padding is
supported; output must be contiguous and disjoint from inputs. Merged, swapped
merged, and separate gate/up tensors are supported. Rejected cases use the
original GGML operator. CPU placement and the time MLP remain unchanged.

This is a single GEGLU replacement. GGML's original GEGLU is already a single
kernel; this patch does not claim to fuse two prior launches. FP8 quantization,
other GEMM shapes, attention, RoPE and KV paths are outside this increment.

## Enable and disable

Build with `GGML_CUDA=ON`, `GGML_CUDA_FLASHRT=ON`,
`CMAKE_CUDA_ARCHITECTURES=89`. The existing CUDA source glob includes the new
file after reconfiguration. `unary.cu` invokes the adapter under the existing
compile-time FlashRT guard.

Both `GGML_CUDA_FLASHRT=1` and `GGML_CUDA_FLASHRT_GEGLU=1` are required at process
startup. GEGLU is off by default; setting the latter to 0 restores GGML GEGLU
while leaving the other accepted FlashRT features configured as before.
`GGML_CUDA_FLASHRT_TRACE=1` records actual dispatch and stream capture state.

## Verification

Local checks completed using CUDA 13.0.88 / GCC 13.3:

- SM89 object compilation of the new kernel and modified unary dispatch.
- Full C++ fixture object compilation with warnings treated as errors.
- CPU-only fixture executable linked against the pinned ggml-base: 54 host
  metadata/address/alias checks passed.

GPU execution is performed by `scripts/run-sm89-geglu-port.py` on the user's
4090. No local GPU or model execution is claimed. The runner preserves the
old CPU-placement H configuration and only toggles the GEGLU switch:

1. Build server and GEGLU fixtures; run host contracts.
2. Twelve fixtures in each of four independent mode/Graph processes: both
   dtypes, decoder/encoder widths, merged/swapped/padded inputs and fallback.
   Each fixture computes eight times, updates inputs, then computes three
   more times. Repeats and same-mode cross-Graph raw bytes must match.
3. Independent double mathematical reference bounds fixed before execution:
   F32 `2e-5*(1+abs(reference))`; F16 `2e-3*(1+abs(reference))`.
4. Three existing model input cases, ten requests per mode/Graph process.
   GEGLU-off must match the saved H baseline. Actual F32 decoder and encoder
   dispatch is required. GEGLU-on model drift is recorded separately, and
   same-mode Graph toggling and repeats must remain exact.

Native FlashRT expf and GGML tanhf are mathematically equivalent formulations
with different floating-point rounding. Model drift is never converted into
a task-accuracy pass by adjusting a threshold after seeing outputs. If drift
is present, the report says `drift_recorded_task_accuracy_not_accepted`; the
new feature remains opt-in. No end-to-end speedup is claimed by this runner.
