# SM89 / pi0 progress checkpoint

This checkpoint records the user-supplied gpu4 results and the scopes enforced
by the delivered runners. The development environment has not directly opened
the remote report files. It does not claim complete FlashRT coverage.

| Region | Integration / numerical evidence | Remaining acceptance |
| --- | --- | --- |
| LayerNorm + Gemma RMSNorm | GGML-math fusion, CUDA fixtures, node shadow and three-input model comparisons passed | Native FP16/FP8 arithmetic is a different path; broad task coverage remains |
| Short-sequence GEMM | GEMM=2, five `(K,N,M=11)` shapes, model outputs matched the earlier baseline across Graph settings | Native provenance/precision distinctions must be retained; timing had substantial variability |
| Bias/residual | Direct and CONT fusion, fixtures and model comparisons matched accepted GEMM | Stable incremental performance benefit not established |
| Bias/activation | GELU CUDA hits; SiLU was placed on CPU in original graph | GPU placement changes are separate and not accepted; do not count CPU SiLU as CUDA coverage |
| Native GEGLU adaptation | Encoder/decoder hits, fixtures, repeatability and cross-Graph agreement | Stable action drift; task accuracy and performance pending; opt-in |
| FP16 prefill / vision GEMM | 12 shapes, GPU fixtures, actual model hits/capture, repeatability and cross-Graph agreement | Stable action drift; task accuracy and performance pending; opt-in |
| Patch IM2COL | Supplied by the accompanying new increment | GPU/reference/model checks pending execution at this checkpoint |
| Patch convolution GEMM + bias/position | Existing host path retained | Remaining native coverage must be assessed separately from IM2COL |
| QKV split / RoPE / KV writes | Existing host path retained | FlashRT adaptation and mask/cache/layout tests pending |
| ViT / encoder / decoder attention | Existing GGML backend retained | FlashRT backend integration and numerical/performance acceptance pending |
| Native FP8 weights, scales, GEMM, norm/GEGLU-to-FP8 | Not implemented by the current F16 GGUF adapters | Managed packing, calibration inputs, scales, lifetimes and validation pending |
| Final norm, action projection / update | Existing host boundaries retained | Native path mapping and acceptance pending |
| Combined full configuration | Individually tested settings exist | All intended kernels together, representative task inputs, accuracy and reliable performance pending |

## Latest completed prefill GEMM stage

Report: `results/flashrt-sm89/prefill-gemm-obwek26o/summary.json`.
Status: `prefill_gemm_integrated`.
Model numerics: `drift_recorded_task_accuracy_not_accepted`.
GEGLU was disabled. Initial noise was unchanged. Native prefill candidate
outputs were repeatable and the same across Graph settings; feature-off
outputs matched the accepted baseline.

| Input case | Final action max absolute difference | RMSE | Relative L2 |
| --- | ---: | ---: | ---: |
| fixed | 0.0158683062 | 0.00438272356 | 1.061189% |
| changed-text | 0.0329561234 | 0.00749130701 | 1.331620% |
| changed-state | 0.00916075706 | 0.00223793566 | 0.465801% |

The native FlashRT FP16 Lt path requests FP32 compute and F16 core output;
the original SM89 GGML path has a different compute configuration. This is
a plausible contributor to the differences, not a proof that every difference
is harmless rounding or that task accuracy is acceptable.

The GEMM fixture reference checks 96 independent FP64 dot-product coordinates
per case and phase; it is not a full-tensor error proof. All outputs are checked
for finiteness and fallback outputs for binary equality. The model-stage final
status requires all 12 intended shapes to hit and, when Graphs are enabled,
each shape to be captured. TRACE counts are not time fractions.

The next independent increment uses PREFILL_GEMM=0 and GEGLU=0, preserving
the prior exact baseline so one candidate's drift does not obscure another's.
No existing result is reclassified as a task-accuracy or performance success.
