# PI0.5 persistent-session regression on Jetson Thor

This documents local validation of the fix in `src/llama-context.cpp`, based on
GitHub master `bbc0f93770f6cbc6dc75685ac2d1cb93e45e99ea`. This is not a comprehensive correctness
certification and a fresh EagleVLA-Edge server is not an OpenPI reference.

## Failure and fix

When the token count changes, `pi0_refresh_encoded_kv_gpu` must reallocate
GPU-resident cross-KV tensors. The original code reset the current encoder
result before copying it into the new GPU buffers. The source vector and its
tensors belong to that result, so they can become empty or invalid. A
persistent decoder can also retain references into the old allocation. This
can corrupt subsequent predictions even with `/foreground/reset`; the
non-persistent path can be affected too.

The fix keeps the live encoder result intact. Before freeing an old KV
allocation, it synchronizes and invalidates the separate persistent decoder
graph and clears the old cross-buffer pointers. Same-shape requests can still
reuse the persistent graph; a changed allocation forces it to be rebuilt.

## Test setup

- Jetson Thor, PI0.5, two 224x224 inputs, 10 diffusion steps, 10x32 actions.
- Model: `/home/nvidia/yzb/jetsonpi_apxinf_thor_20261001/Jetson-PI-Edge/devlocal/thor-perf/models/Pytorch_Model-2.9B-Thor-mixed-10action.gguf`.
- Vision model: same directory, `mmproj-model-matmul-bf16.gguf`.
- Noise: `/home/nvidia/yzb/jetsonpi_apxinf_thor_20261001/pi0_noise_seed41_10x32.bin`.
- Input: `/home/nvidia/yzb/jetsonpi_apxinf_thor_20261001/test-224.jpg`.
- Each of six inputs is first run on its own fresh, non-persistent server.
  A single server with `PI0_PERSISTENT_CUDA_GRAPH=1` then processes the inputs
  in repeated A -> changed input -> A sequences, with a foreground reset before
  each request. Variations include instruction length, second image, and state.
- Acceptance for each 10x32 result: all values finite, cosine similarity >
  0.999 and max absolute error <= 0.01 versus its fresh-server result.

Run the regression from the repository root (choose an unused output directory
and port):

```bash
python3 tests/test-pi05-foreground-persistent.py \
  --server /path/to/build-thor/bin/llama-server \
  --model /path/to/Pytorch_Model-2.9B-Thor-mixed-10action.gguf \
  --mmproj /path/to/mmproj-model-matmul-bf16.gguf \
  --noise /path/to/pi0_noise_seed41_10x32.bin \
  --image /path/to/test-224.jpg \
  --out /path/to/new-output-directory --port 18175 --rounds 3
```

This uses NVIDIA GPU 0 by default (`--gpu` selects another CUDA device), loads
and unloads six reference servers and then one
candidate server, and stores server logs, raw actions, references and a JSON
summary under the output directory. It requires `numpy` and Pillow. It uses
the existing Python foreground HTTP client.

## Results and limitations

The historical unmodified server failed 6 of 11 comparisons over one round:
minimum cosine 0.4922, maximum absolute difference 4.1044. The fixed build
passed 33 of 33 comparisons across three rounds: minimum cosine
0.9999999984, maximum absolute difference 0.0005563. The original default
non-persistent server also produced non-finite values in a separate
instruction-switch reproduction. These results establish a real bug in that
scenario, not that every prediction from GitHub is wrong.

For 50 steady-state requests following 10 warmups, the fixed persistent run
had P50 model-stage sum 203.011 ms, `/infer` 204.423 ms, and full client
sequence (reset, two image submissions, state, infer) 207.476 ms. The fixed
non-persistent run gave 204.096 ms, 205.483 ms, and 208.363 ms respectively.
Model-stage sum is ViT + LLM + action-expert timing per request, not network
end-to-end latency. Nsight captured 13 executions of one persistent CUDA graph
for the stable-shape action expert.

The 33-case regression above compares to the same implementation on a fresh
server, not to OpenPI. It does not independently prove correctness of the
model conversion, preprocessing, all input shapes, or other models and
hardware.

## Additional OpenPI comparison

A separate two-image PI0.5 comparison used the local OpenPI PyTorch code and
`/data/home/yzb/model/pi05_base/model.safetensors`, with the F16 GGUF model and
vision model at `/data/home/yzb/model/gguf/pi05_base/` on Thor. The GGUF action
projection bias and vision patch bias were checked against the checkpoint;
this is not a tensor-by-tensor verification of the entire conversion. OpenPI
used its model's mixed precision and ran on an RTX 4090, whereas the GGUF ran
on Thor. Two copies of the same JPEG, state, prompt tokens, 10x32 fixed noise,
10 diffusion steps, and the raw action output stage were aligned. Image and
noise files were SHA-256 checked across machines. The valid text-token
sequences, including BOS, matched for both prompts.

| Output | Fresh server vs OpenPI cosine | Persistent server vs OpenPI cosine | Maximum absolute error, persistent vs OpenPI |
|---|---:|---:|---:|
| A: `put both moka pots on the stove` | 0.999452783 | 0.999453128 | 0.0152479 |
| B: `move the mug to the other side of the table` | 0.999916550 | 0.999916550 | 0.0149015 |

Both exceed the guide's cosine threshold of 0.999. All 21 A/B outputs across
three repeated mixed-input sequences were compared directly to OpenPI; their
minimum cosine was 0.999453128. These OpenPI absolute differences are larger
than the separate 0.01 **fresh-server regression** tolerance; that tolerance
was applied between EagleVLA server configurations, not between different
frameworks and precisions. The same F16 GGUF also passed all 33 fresh-server
comparisons; its worst cosine was 0.9999999337 and maximum absolute difference
was 0.0003195.

An additional RTX 4090 check rebuilt the FlashRT C provider from this source
and ran the PI0 and PI0.5 server/provider paths against OpenPI for the same
two prompts and two images. For PI0, server/OpenPI and provider/OpenPI cosine
were 0.9999717 (A) and 0.9999358 (B). For PI0.5 they were 0.9997087 (A) and
0.9998878 (B). The provider exactly matched the fresh-server outputs for
seven alternating A/B calls in both models. PI0 unroll and PI0.5 persistent
server sessions each passed 33 fresh-server comparisons. Different precision
and device builds may produce small output differences.

The local OpenPI source used in this check contains experimental hard-coded
noise and state overrides in `sample_actions`. The PI0 reference script
removed those two overrides in memory without modifying that source, so the
actual observation and provided noise were used. The 320 hard-coded noise
values in the PI0.5 OpenPI run were independently checked to equal the fixed
noise file; PI0.5 state is encoded in the prompt tokens before that override.

This is evidence for these inputs, not exhaustive exact-precision parity,
three-camera testing, exhaustive tensor verification, or robot task tests.
