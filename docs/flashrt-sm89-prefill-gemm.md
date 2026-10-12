# FlashRT FP16 encoder / vision GEMM increment

This increment adds an opt-in SM89 adapter of FlashRT `GemmRunner::fp16_nn`
to GGML `MUL_MAT`. It extends the existing five short-sequence GEMM shapes;
it does not replace their `GGML_CUDA_FLASHRT_GEMM=2` implementation.

## Source and arithmetic

Source: [FlashRT, commit e881e142](https://github.com/flashrt-project/FlashRT/blob/e881e1424cf4fadee7dff121d01cc4e159e550e7/csrc/gemm/gemm_runner.cu),
`FP16_NN` descriptor setup and `GemmRunner::fp16_nn`. The non-FP8 encoder
branch of `flash_rt/models/pi0/pipeline_rtx.py` calls this path.
The complete Apache-2.0 license and adaptation notes are under
`vendor/flashrt/prefill-gemm/`.

The native core is a cuBLASLt GEMM with **F16 operands, FP32 compute and F16
output**. It is not a new handwritten matrix multiplication kernel. The GGML
adapter explicitly rounds F32 activation inputs to F16, then extends the F16
core output to F32. For native F16 activation inputs the first conversion is
omitted. Existing GGML weights are used without a transpose/repacking allocation:
`W[n*K+k]`, `X[m*K+k]`, `D[m*N+n]` are described as column-major `op(W)=T`,
`op(X)=N`. Descriptor layout and algorithm selection differ from the original
runtime; identical arithmetic precision does not imply bitwise agreement.

At pinned Jetson commit `1271fa933c3f81e1cb3a71a9f09f13bdefa4461b`, SM89's
default large F16-weight GGML path requests F16 compute. Consequently this
candidate can produce model action differences. It must not be declared
task-accuracy accepted merely because outputs are finite or repeatable.

## Exact dispatch coverage

Dimensions below are GEMM dimensions, with GGML weights `ne=(K,N)`.

| Domain | M | K,N pairs |
| --- | --- | --- |
| Vision | 256 | (1152,1152), (1152,4304), (4304,1152), (1152,2048) |
| Encoder | 264 | (2048,2048), (2048,256), (2048,16384), (16384,2048) |
| Encoder | 268 | (2048,2048), (2048,256), (2048,16384), (16384,2048) |

Both `GGML_CUDA_FLASHRT=1` and `GGML_CUDA_FLASHRT_PREFILL_GEMM=1` are required.
Unset or `0` keeps existing dispatch. Unsupported shapes, M=2, M=11,
non-SM89 devices, tensor layouts/types/aliases, explicit precision or hints,
and any explicit `GGML_CUDA_CUBLAS_COMPUTE_TYPE` override return to existing
GGML dispatch before writes. The weight must be a non-view F16 tensor in a
CUDA buffer marked `WEIGHTS`; all operands must be contiguous, aligned,
nonoverlapping, and inside same-device CUDA buffer allocations.

M=2 is deliberately excluded. The server's sequence-deletion capability
probe calls `llama_decode` with two zero tokens during initialization,
independently of `--no-warmup`. Those TRACE events do not establish normal
request coverage. M=264/268 were observed for this session's fixed input cases;
other prompt lengths intentionally fall back until separately covered.

## CUDA Graph and resource lifetime

Each context/stream owns its Lt handle, twelve descriptor/algorithm entries,
32 MiB workspace, and two F16 scratch allocations of `268*16384` elements.
This is approximately 48.75 MiB per used stream. Upstream's default workspace
is 256 MiB; this adapter's smaller cap is an explicit engineering change.
There is no autotuning or algorithm search during graph capture.

Plans are created before capture. A shape must have executed outside capture
before its captured candidate path is enabled. Unsupported heuristics or an
unprepared capture return to GGML. After an enqueue, CUDA/Lt failures are
reported as errors, never hidden by a fallback. Context destruction resets
these states while holding GGML's capture exclusion lock, synchronizes the
device, then frees scratch/descriptors/handles before destroying streams.

## Verification delivered with this increment

`test-flashrt-prefill-gemm` includes 68 host metadata/address checks, plus
17 GPU cases (12 primary F32-input shapes, one native F16-input case, and
four fallback cases). Each GPU case runs eight initial computations and
three computations after changing input values at the same addresses.

Every output element must be finite. Independent double-precision dot products
check 96 dispersed output coordinates per case and phase using the actual
F16-rounded operands. The explicit-F32-precision fallback uses actual F32 X.
The preset bound is `0.004 + 0.002*abs(reference)`; this is an engineering
test threshold, not a proven bound on arbitrary FP16 reductions or model inputs.
It gates only the new eligible candidate core. Original GGML and fallback
paths retain their actual reference errors and out-of-bound sample counts;
the candidate's four fallback cases must instead match original GGML bytes
exactly. This prevents attributing existing FP16 accumulation error to the
new adapter, without relaxing the candidate threshold.
The sampled reference is **not** a full-tensor accuracy proof. Same-mode
repeated outputs and Graph-on/off binary outputs must match exactly.

`scripts/run-sm89-prefill-gemm.py` builds the increment, pins one available
4090 UUID, executes four independent fixture processes, then four model
processes with fixed / changed-text / changed-state inputs. It checks actual
candidate TRACE and capture coverage; a fallback-only run cannot pass.
The model baseline preserves accepted norm, GEMM=2 and bias/residual settings,
original CPU time-MLP placement, and disables GEGLU and bias/activation.
Only the new prefill GEMM switch changes between model groups.

The existing GEGLU stage's mode-0 model outputs remain the reference. Native
GEMM action differences are recorded without automatically accepting task
accuracy. Input noise, repeatability and cross-Graph consistency are checked
separately. Only children created by this invocation are stopped. Existing
results and processes are retained. No performance measurement is included;
TRACE counts are not time fractions or evidence of speedup.

## Current completion boundary

Local CUDA 13.0 / SM89 object compilation and host checks can be completed
without a GPU. GPU fixtures and model stages must pass on the user's 4090.
This increment does not finish FlashRT's native FP8 weight/scaling and
quantization pipeline, QKV/RoPE/KV handling, attention, or the action update
path. It does not validate robot task accuracy or overall performance.
