<div align="center">
  <p>
    <a href="https://www.pku.edu.cn/"><img src="media/pku-wordmark-red.png" alt="Peking University" width="300" /></a>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://www.tlaic.ac.cn/"><img src="media/tlaic-wordmark.svg" alt="Beijing Tongminghu Information Technology Application Innovation Center" width="190" /></a>
  </p>

  <img src="media/jetson-pi-edge-logo.svg" alt="Jetson-PI Edge" width="720" />

  <p><strong>A llama.cpp-based inference engine for real-time onboard Vision-Language-Action control.</strong></p>
  <p>Run VLA locally on NVIDIA Jetson, desktop GPUs, CPUs, NPUs, and other edge platforms through a persistent robot-facing HTTP runtime.</p>

  <p>
    <a href="https://arxiv.org/abs/2607.12659"><img src="https://img.shields.io/badge/arXiv-2607.12659-b31b1b.svg" alt="arXiv" /></a>
    <a href="https://www.corl.org/"><img src="https://img.shields.io/badge/CoRL-2026_Accepted-6f42c1.svg" alt="Accepted at CoRL 2026" /></a>
    <a href="https://github.com/PKU-SEC-Lab/Jetson-PI"><img src="https://img.shields.io/badge/Jetson--PI-Algorithm-35b8a9.svg" alt="Jetson-PI algorithm" /></a>
    <a href="https://huggingface.co/diantoudefengshan/Jetson-PI-GGUF"><img src="https://img.shields.io/badge/Model-GGUF-f5c542.svg" alt="GGUF checkpoints" /></a>
    <a href="https://github.com/ggml-org/llama.cpp"><img src="https://img.shields.io/badge/Based_on-llama.cpp-3578c8.svg" alt="Based on llama.cpp" /></a>
    <a href="https://github.com/flashrt-project/FlashRT"><img src="https://img.shields.io/badge/Support-FlashRT-f28c45.svg" alt="Support FlashRT" /></a>
  </p>

  <p>
    <a href="#real-world-demo">Demo</a> ·
    <a href="#quick-start">Quick Start</a> ·
    <a href="#gr00t-n17-c-api-and-server">GR00T APIs</a> ·
    <a href="#performance-on-jetson-orin">Performance</a> ·
    <a href="docs/model_conversion.md">Model Conversion</a> ·
    <a href="docs/foreground_server_usage.md">Server API</a> ·
    <a href="#citation">Citation</a>
  </p>
</div>

---

## News

- **[2026/09] 🎉🎉🎉 Jetson-PI has been accepted as a Spotlight at CoRL 2026!**
- **[2026/08] Pre-converted GGUF checkpoints are available.** Ready-to-use PI0, PI0.5, and GR00T N1.7 main-model and vision-projector GGUF files are released on [Hugging Face](https://huggingface.co/diantoudefengshan/Jetson-PI-GGUF).
- **[2026/08] NVIDIA Isaac GR00T N1.7 inference is available.** Convert the official checkpoint to GGUF and run the complete Qwen3-VL backbone, Action Head, and four-step action flow through either `libjetson_pi_gr00t` or the foreground HTTP server.
- **[2026/07] Python APIs are available.** Use the managed foreground NumPy client for persistent `llama-server` sessions or the optional pybind11 module for in-process PI0/PI0.5 action inference.
- **[2026/07] Updated to the latest llama.cpp codebase.** Jetson-PI-Edge now tracks the latest llama.cpp architecture while retaining PI0/PI0.5 inference, foreground server, and FlashRT integration support.
- **[2026/07] FlashRT support is available.** Jetson-PI is exposed as a FlashRT-loadable provider through a C API, allowing [FlashRT](https://github.com/flashrt-project/FlashRT) to invoke the same PI0/PI0.5 model path directly from Python without starting the foreground HTTP server.
- **[2026/07] Jetson-PI is open source.** We released the [Jetson-PI asynchronous control framework](https://github.com/PKU-SEC-Lab/Jetson-PI) and this [Jetson-PI-Edge inference engine](https://github.com/PKU-SEC-Lab/Jetson-PI-Edge).

## About

Jetson-PI-Edge is the inference engine accompanying our paper:

> **[Jetson-PI: Towards Onboard Real-Time Robot Control via Foresight-Aligned Asynchronous Inference](https://arxiv.org/abs/2607.12659)**<br>
> Zebin Yang, Qi Wang, Yunhe Wang, Xiurui Guo, Bo Yu, Shaoshan Liu, Jiafeng Xu, Hao Dong, and Meng Li.<br>
> Accepted at the Conference on Robot Learning (CoRL), 2026.

Deploying VLA policies on low-power onboard hardware is difficult because model latency directly limits control frequency and responsiveness. Jetson-PI combines foresight-aligned asynchronous correction with confidence-aware scheduling and system-level acceleration. This repository provides the llama.cpp-based execution layer: GGUF model loading, multimodal encoding, PI and GR00T action inference, graph reuse, and robot-facing runtime interfaces.

The project is built on [llama.cpp](https://github.com/ggml-org/llama.cpp). The asynchronous control algorithm lives in [PKU-SEC-Lab/Jetson-PI](https://github.com/PKU-SEC-Lab/Jetson-PI).

## Real-World Demo

<p align="center">
  <img src="video/demo.gif" alt="Jetson-PI real-world comparison demo" width="720" />
</p>

<p align="center">
  <a href="video/demo.mp4">▶ Watch the full-resolution demo (MP4)</a>
</p>

## Current Support and Roadmap

### Supported Models

- ✅ **PI0** - PI-specific multimodal preprocessing, language backbone, action expert, and continuous action-chunk generation.
- ✅ **PI0.5** - two-view prefix construction, PI0.5 prompt/state handling, and 10-step action generation.
- ✅ **NVIDIA Isaac GR00T N1.7** - Qwen3-VL multimodal backbone, complete GR00T Action Head, four-step flow matching, and `[40, 132]` action generation.
- ✅ **Automatic model dispatch** - detect PI0, PI0.5, and GR00T N1.7 from GGUF metadata and model tensor names, with an explicit `PI_MODEL` override for PI models.

### Runtime Optimizations

- ✅ **Graph reuse** - reuse the ViT, language backbone, and action-expert computation graphs across inference rounds.
- ✅ **GPU-resident intermediate buffers** - keep reusable embeddings and cross-KV intermediates on the accelerator to reduce host/device traffic.
- ✅ **Flow unrolling** - execute multiple action-expert flow steps in one graph to reduce scheduling and launch overhead.

### Interfaces and Deployment

- ✅ **Foreground HTTP server** - persistent PI0, PI0.5, and GR00T image, robot-state, and instruction endpoints with action and timing outputs.
- ✅ **FlashRT provider** - expose the same PI0/PI0.5 GGUF runtime through a C API for the FlashRT Python model interface.
- ✅ **GR00T C API** - accept raw multi-camera RGB, instruction, normalized state, and embodiment ID through `libjetson_pi_gr00t`.

The foreground HTTP server supports PI0, PI0.5, and GR00T N1.7. The managed Python
foreground client and FlashRT provider currently remain specific to PI0 and PI0.5;
GR00T callers can use the HTTP endpoints directly or the standalone C API.

### Planned Model Support

#### Vision-Language-Action Models

- [ ] [LingBot-VLA 2.0](https://github.com/Robbyant/lingbot-vla-v2) - cross-embodiment VLA covering manipulators and humanoid robots.
- [ ] [Qwen-RobotManip](https://qwen.ai/blog?id=qwen-robotmanip) - Qwen-based manipulation VLA with representation, motion, and behavior alignment.

#### World-Action and World-Model-Driven Models

- [ ] [DreamZero](https://github.com/dreamzero0/dreamzero) - world-action model that jointly predicts future video and robot actions.
- [ ] [FastWAM](https://github.com/yuantianyuan01/FastWAM) - real-time WAM that uses video co-training while skipping future imagination at inference time.

## Runtime Architecture

```mermaid
flowchart LR
    CAM[Camera views] --> SESSION[Foreground session]
    STATE[Robot state] --> SESSION
    TEXT[Language instruction] --> SESSION
    SESSION --> VIT[SigLIP or Qwen3-VL mmproj]
    VIT --> VLM[PI or GR00T backbone]
    VLM --> AE[PI action expert or GR00T Action Head]
    AE --> ACTION[Action chunk]
    ACTION --> ROBOT[Robot controller]

    CACHE[Graph reuse + GPU buffers] -. accelerates .-> VIT
    CACHE -. accelerates .-> VLM
    CACHE -. accelerates .-> AE
```

The server keeps the foreground session alive across requests. A control process uploads sensor observations, calls inference once, and consumes a flat action tensor plus detailed latency fields. This avoids rebuilding the model interface for every control step.

## Quick Start

### 1. Build the server

CPU-only:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server -j
```

NVIDIA Jetson or another CUDA-capable target:

```bash
cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server -j
```

### 2. Prepare the model

The runtime expects two matching files:

- a PI language/action model in GGUF format;
- a SigLIP vision encoder/projector in GGUF format.

Convert a PI0 or PI0.5 checkpoint by following [Model Preparation](https://github.com/PKU-SEC-Lab/Jetson-PI-Edge/blob/master/docs/model_conversion.md), or download the ready-to-use GGUF checkpoints directly from [Hugging Face](https://huggingface.co/diantoudefengshan/Jetson-PI-GGUF).

### 3. Start the foreground server

```bash
PI_MODEL=auto \
./build/bin/llama-server \
  -m /path/to/pi_llm.gguf \
  --mmproj /path/to/mmproj.gguf \
  -ngl 99 \
  --host 0.0.0.0 \
  --port 8080
```

Use `PI_MODEL=pi0` or `PI_MODEL=pi05` to force a model path.

**RTX 4090 (single GPU):**

```bash
CUDA_VISIBLE_DEVICES=0 \
PI_MODEL=auto \
./build/bin/llama-server \
  -m /path/to/pi_llm.gguf \
  --mmproj /path/to/mmproj.gguf \
  -ngl 37 \
  --chat-template vicuna \
  --host 127.0.0.1 \
  --port 8080 \
  -np 1 \
  -t 8 \
  -tb 8
```

The `-t 8 -tb 8` settings configure the host thread pools used by the persistent foreground server and are recommended for the tested RTX 4090 setup.

### 4. Run one inference round

Reset the persistent session:

```bash
curl -X POST http://127.0.0.1:8080/foreground/reset
```

Submit two camera views:

```bash
curl -X POST http://127.0.0.1:8080/foreground/image \
  -H 'Content-Type: application/json' \
  -d '{"path":"/path/to/image_1.png"}'

curl -X POST http://127.0.0.1:8080/foreground/image \
  -H 'Content-Type: application/json' \
  -d '{"path":"/path/to/image_2.png"}'
```

Submit the robot state:

```bash
curl -X PUT http://127.0.0.1:8080/foreground/state \
  -H 'Content-Type: application/json' \
  -d '{"state":"1.8731,-1.0370,1.9652,7.0876,0.2546,-9.1432,-0.0147,-0.5037,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"}'
```

Request an action chunk:

```bash
curl -X POST http://127.0.0.1:8080/foreground/infer \
  -H 'Content-Type: application/json' \
  -d '{"text":"pick up the object and place it into the tray"}'
```

The response contains `action_final` together with model and server timing fields such as `encode_ms`, `decode_ms`, `total_ms`, and `timing_breakdown_ms`. See the [Foreground Server API](docs/foreground_server_usage.md) for endpoint semantics and all response fields.

## GR00T N1.7 C API and Server

GR00T N1.7 is available through both a standalone product library and the existing
foreground HTTP server. Build the runtime with CUDA support using:

```bash
cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target jetson_pi_gr00t llama-server -j
```

Download the ready-to-use BF16 [GR00T N1.7 main GGUF](https://huggingface.co/diantoudefengshan/Jetson-PI-GGUF/resolve/main/gr00t-n1d7/gr00t-n1d7-bf16.gguf)
and matching [vision-projector GGUF](https://huggingface.co/diantoudefengshan/Jetson-PI-GGUF/resolve/main/gr00t-n1d7/mmproj-gr00t-n1d7-bf16.gguf),
or convert the official GR00T and Cosmos checkpoints by following the
[GR00T validation and conversion guide](tools/gr00t-reference/README.md). Always use
the main model and mmproj from the same release directory.
Applications include `jetson_pi_gr00t.h`, link against `libjetson_pi_gr00t`, and use:

```c
jetson_pi_gr00t_config config = {
    .struct_size   = sizeof(jetson_pi_gr00t_config),
    .model_path    = "/path/to/gr00t-n1d7.gguf",
    .mmproj_path   = "/path/to/mmproj-gr00t-n1d7.gguf",
    .backend       = "cuda",
    .n_images      = 4,
    .image_height  = 180,
    .image_width   = 320,
    .n_threads     = 0,
    .embodiment_id = 24,
};

jetson_pi_gr00t * engine = NULL;
if (jetson_pi_gr00t_open(&config, &engine) != JETSON_PI_GR00T_OK) {
    /* Handle jetson_pi_gr00t_open_error(). */
}
```

Call `jetson_pi_gr00t_infer()` for one complete policy tick. It accepts RGB-interleaved
raw images in the checkpoint's camera/time order, a UTF-8 instruction, and an already
normalized state. The library performs GR00T image preprocessing, Qwen3-VL multimodal
inference, the complete Action Head, and four flow-matching steps before returning the
checkpoint's normalized `[40, 132]` action chunk. Query the exact output shape with
`jetson_pi_gr00t_action_shape()`.

To use Server mode, start `llama-server` with the GR00T main GGUF and matching mmproj.
The server detects `gr00t-n1d7` from model metadata; do not set `PI_MODEL`:

```bash
./build/bin/llama-server \
  -m /path/to/gr00t-n1d7.gguf \
  --mmproj /path/to/mmproj-gr00t-n1d7.gguf \
  -ngl 99 --host 0.0.0.0 --port 8080
```

Use the normal `/foreground/reset` and ordered `/foreground/image` calls, then submit
the normalized state together with its GR00T embodiment ID:

```bash
curl -X PUT http://127.0.0.1:8080/foreground/state \
  -H 'Content-Type: application/json' \
  -d '{"state":[0.0,0.0,0.0],"embodiment_id":24}'

curl -X POST http://127.0.0.1:8080/foreground/infer \
  -H 'Content-Type: application/json' \
  -d '{"text":"pick up the object and place it into the tray"}'
```

The server pads short state arrays to the width stored in GGUF metadata and returns the
normalized action in `action_final` with `action_steps` and `action_dim`. Image upload
order must match the checkpoint's camera/time convention. The managed Python foreground
client and FlashRT provider do not yet dispatch GR00T models.

## Python Foreground API

The Python foreground client can start `llama-server`, wait for the model to become ready, submit images and state through the persistent `/foreground/*` session, and return the action as a NumPy array.

### Build llama-server

```bash
cmake -S . -B build-graph \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DLLAMA_BUILD_SERVER=ON

cmake --build build-graph --target llama-server -j8
```

Install NumPy and make the source-tree client importable:

```bash
python3 -m pip install numpy
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
```

### Start and call the server from Python

```python
import numpy as np

from jetson_pi_foreground import ManagedForegroundSession

SERVER_PATH = "/path/to/Jetson-PI-Edge/build-graph/bin/llama-server"
MODEL_PATH = "/path/to/pi0-model.gguf"
MMPROJ_PATH = "/path/to/mmproj-model.gguf"
IMAGE_PATH = "/path/to/test-224.jpg"

state = np.asarray([
    -1.8731, -1.0370, 1.9652, 7.0876,
     0.2546, -9.1432, -0.0147, -0.5037,
] + [0] * 24, dtype=np.float32)

# This starts llama-server and waits for /health to report ready.
session = ManagedForegroundSession(
    server_path=SERVER_PATH,
    model_path=MODEL_PATH,
    mmproj_path=MMPROJ_PATH,
    gpu=0,
    port=8080,
    timeout=300,
)

try:
    action, metadata = session.predict(
        image_paths=[IMAGE_PATH, IMAGE_PATH],
        prompt="/do something",
        state=state,
        reset=True,
    )
    np.savetxt("action.txt", action, fmt="%.9g")
    print("action shape:", action.shape)
    print("total_ms:", metadata.get("total_ms"))
finally:
    session.close()
```

Keep the same `session` alive for repeated control steps so that the model and CUDA context are loaded only once. Use `reset=True` for the first step or when starting a new foreground session, and call `session.close()` when the managed server is no longer needed.

## FlashRT Support

Jetson-PI can be loaded by [FlashRT](https://github.com/flashrt-project/FlashRT) through a C API provider. The provider reuses the same llama.cpp-based PI0/PI0.5 implementation and GGUF model path from this repository, while FlashRT supplies the Python-facing model API. The foreground HTTP server is not required for this integration.

After installing FlashRT, configure its C++ build with this repository as the Jetson-PI source tree:

```bash
cmake -S /path/to/FlashRT/cpp -B /path/to/FlashRT/cpp/build-jetson-pi \
  -DCMAKE_BUILD_TYPE=Release \
  -DFLASHRT_CPP_WITH_JETSON_PI=ON \
  -DJETSON_PI_ROOT=/path/to/Jetson-PI-Edge \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_FA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=<target-sm>

cmake --build /path/to/FlashRT/cpp/build-jetson-pi \
  --target flashrt_cpp_llama_cpp_provider_c -j
```

The build produces `libflashrt_cpp_llama_cpp_provider_c.so`. Load it from Python with `framework="jetson_pi"`:

```python
import os

import flash_rt
import numpy as np
from PIL import Image


def load_rgb224(path):
    image = Image.open(path).convert("RGB")
    if image.size != (224, 224):
        image = image.resize((224, 224), Image.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


model = flash_rt.load_model(
    "/path/to/pi_llm.gguf",
    framework="jetson_pi",
    config="pi0",
    mmproj_path="/path/to/mmproj.gguf",
    backend="cuda",
    num_views=2,
    action_steps=10,
    action_dim=32,
    # Usually produced at:
    # <FlashRT repo>/cpp/<build-dir>/libflashrt_cpp_llama_cpp_provider_c.so
    lib_path="/path/to/FlashRT/cpp/build-jetson-pi/libflashrt_cpp_llama_cpp_provider_c.so",
)

image = load_rgb224("/path/to/image.png")
images = [image, image]

state = np.asarray([
    -1.8731, -1.0370, 1.9652, 7.0876, 0.2546, -9.1432, -0.0147, -0.5037,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
], dtype=np.float32)

prompt = "do something"
actions = model.predict(images=images, prompt=prompt, state=state)

np.savetxt("actions_10x32.txt", np.asarray(actions, dtype=np.float32), fmt="%.9g")
```

See the [FlashRT repository](https://github.com/flashrt-project/FlashRT) for installation and its complete API. The standalone llama.cpp foreground server remains available independently and does not require FlashRT.



## Performance on Jetson Orin

Latency is measured in milliseconds on NVIDIA Jetson Orin in MAXN mode. See the paper for the complete experimental setup and end-to-end control results.

### PI0

| Runtime configuration | ViT | LLM | Action Expert | Total |
|---|---:|---:|---:|---:|
| Naive PI0 | 143.5 | 601.9 | 505.5 | 1250.9 |
| + Schedule optimization | 147.1 | 603.1 | 501.3 | 1251.5 |
| + Graph reuse | 76.8 | 200.6 | 167.0 | 444.4 |
| + Intermediate buffer and unroll | **75.4** | **200.3** | **118.8** | **394.5** |

### PI0.5

| Runtime configuration | ViT | LLM | Action Expert | Total |
|---|---:|---:|---:|---:|
| Naive PI0.5 | 152.3 | 631.0 | 536.8 | 1420.8 |
| + Schedule optimization | 152.3 | 631.0 | 536.8 | 1420.8 |
| + Graph reuse | 79.5 | 212.6 | 184.0 | 476.1 |
| + Intermediate buffer and unroll | **79.5** | **210.3** | **123.1** | **412.9** |


## Documentation

| Document | Description |
|---|---|
| [Model Preparation](docs/model_conversion.md) | Download, split, and convert PI checkpoints to GGUF. |
| [GR00T Validation and Conversion](tools/gr00t-reference/README.md) | Convert GR00T N1.7 to GGUF and run reference and regression utilities. |
| [Foreground Server API](docs/foreground_server_usage.md) | Session lifecycle, endpoints, response fields, and operational notes. |
| [ViT Optimization](latency_optimize_docs/pi0_vit_optimization.md) | Vision graph reuse and input-layout optimization. |
| [Decode Graph Reuse](latency_optimize_docs/pi0_decode_graph_reuse.md) | Action-expert graph reuse implementation. |
| [GPU KV Cache](latency_optimize_docs/pi0_decode_kv_gpu.md) | GPU-resident cross-KV cache design. |

## Citation

If Jetson-PI or Jetson-PI-Edge helps your research, please cite our paper:

```bibtex
@article{yang2026jetson,
  title={Jetson-PI: Towards Onboard Real-Time Robot Control via Foresight-Aligned Asynchronous Inference},
  author={Yang, Zebin and Wang, Qi and Wang, Yunhe and Guo, Xiurui and Yu, Bo and Liu, Shaoshan and Xu, Jiafeng and Dong, Hao and Li, Meng},
  journal={arXiv preprint arXiv:2607.12659},
  year={2026}
}
```

## Acknowledgments

Jetson-PI-Edge builds on [llama.cpp](https://github.com/ggml-org/llama.cpp), [OpenPI](https://github.com/Physical-Intelligence/openpi), the PI model family from [Physical Intelligence](https://www.physicalintelligence.company/), and [NVIDIA Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T). We also thank the [FlashRT](https://github.com/flashrt-project/FlashRT) project for its high-performance real-time VLA deployment path.
