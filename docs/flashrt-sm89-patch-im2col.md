# FlashRT patch_im2col: GGML layout and dtype adaptation

This increment ports the one-output-element-per-thread gather from
[FlashRT patch_embed.cu](https://github.com/flashrt-project/FlashRT/blob/e881e1424cf4fadee7dff121d01cc4e159e550e7/csrc/kernels/patch_embed.cu)
at commit `e881e1424cf4fadee7dff121d01cc4e159e550e7` into GGML's CUDA
`IM2COL` dispatch. The full Apache-2.0 license is reproduced under
`vendor/flashrt/patch-im2col/`.

It covers image patch extraction only. Convolution/projection GEMM, bias,
position embeddings, image preprocessing, CPU/GPU placement and HTTP behavior
retain their existing implementation.

## Why the layout and dtype adaptation is necessary

FlashRT's original `patch_im2col` reads F16 NHWC images and writes patches
with features ordered `(kh,kw,c)`. Current Jetson-PI pi0 reads F32 CHW images
and needs `(c,kh,kw)` features for its existing convolution weights.
`tools/mtmd/models/siglip.cpp` also promotes the patch kernel to F32 before
calling `ggml_conv_2d`; `ggml.c` selects IM2COL's output dtype from that
kernel. Consequently the actual model path is **F32 to F32**.

The new adapter changes the index decoding and supports GGML F32 input with
F32 or F16 output. It retains FlashRT's flat one-dimensional gather launch
and fixed 14-by-14 patch geometry. F32 output is a direct copy; F16 output
uses round-to-nearest-even conversion. There is no intermediate NHWC buffer,
weight reordering, or F32-to-F16-to-F32 round trip on the F32 path. This is
a layout/dtype adaptation, not a claim of preserving the complete original
FP16 image pipeline or the optional uint8/LUT path.

For one image, the exact mapping is:

```text
output[(ph*16+pw)*588 + c*196 + kh*14 + kw]
    = image[c*50176 + (ph*14+kh)*224 + pw*14+kw]
```

Batch offsets are independent. The tested batch sizes are one and two;
current pi0 uses a singleton view. A legal view of an allocated image is
supported, including a nonzero byte offset.

## Dispatch and execution contract

- Both `GGML_CUDA_FLASHRT=1` and `GGML_CUDA_FLASHRT_PATCH_IM2COL=1` are required.
  The new switch is disabled by default.
- SM89 only; input `(224,224,3,B)`, output `(588,16,16,B)`, `B` in `{1,2}`,
  kernel metadata `(14,14,3,1152)`.
- IM2COL parameters: strides `(14,14)`, padding `(0,0)`, dilation `(1,1)`, 2D.
- Kernel metadata is read for shape only; weight data need not be fetched.
  Actual image/output data must lie in same-device CUDA allocations, with
  exact contiguous strides, proper alignment, valid ranges and no overlap.
- Other shapes, layouts, types, devices and convolution parameters return to
  the existing implementation before writes.
- Execution uses the current GGML CUDA stream. No workspace, plan cache,
  allocator mutation or initialization is needed during capture. CUDA launch
  errors are reported instead of being hidden by fallback.
- `GGML_CUDA_FLASHRT_TRACE=1` emits `flashrt_patch_im2col:` events with output
  dtype, batch, shape, node and capture status. Counts do not measure time.

## Verification

The supplied test uses a CPU coordinate implementation to compare **every
output element bit-for-bit**. It includes the actual F32 model dtype, F16
conversion, two-image batch boundaries, a legal offset view, and four valid
GGML fallback geometries. Input data vary across channels, coordinates and
batches. Each case performs eight initial graph computations followed by
three computations with changed values at the same input addresses.

The runner executes independent feature-off/on and Graph-off/on fixture
processes. Every candidate-eligible fixture must actually hit the new
kernel; every fallback must remain on the existing path. All raw fixture
outputs must match both CPU references and the original GGML implementation,
including comparisons across Graph settings.

The model reference is the **mode-0** group in
`results/flashrt-sm89/prefill-gemm-obwek26o/summary.json`. Accepted norm,
short-sequence GEMM=2 and bias/residual settings are retained. Prefill GEMM,
GEGLU and bias/activation are disabled in both groups; only PATCH_IM2COL
changes. All three input cases must preserve all three action arrays exactly
in both Graph settings. Real model F32/single-image kernel hits and capture
are required. A fallback-only model run cannot pass.

The script preserves earlier reports and only cleans up its own server
processes. It selects and pins one available RTX 4090 for all stages. It does
not measure performance. Completing these checks means patch extraction is
integrated for the stated inputs, not that the entire FlashRT port or robot
task-accuracy evaluation is complete.
