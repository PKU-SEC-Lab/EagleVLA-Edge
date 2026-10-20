# Python NumPy API

## Foreground server session

Use `jetson_pi_foreground.ForegroundSession` when inference must follow the
llama-server `/foreground/*` path. The server owns the persistent model,
CUDA context, KV state, and foreground session. `infer()` returns the raw
action as a contiguous float32 NumPy array together with the full response.

```python
import numpy as np
from jetson_pi_foreground import ForegroundSession

session = ForegroundSession("http://127.0.0.1:8080")
state = np.zeros(32, dtype=np.float32)
action, metadata = session.predict(
    ["view0.jpg", "view1.jpg"],
    "/do something",
    state,
    reset=True,
)
```

For connection reuse, pass `keep_alive=True` to `ForegroundSession` or
`ManagedForegroundSession` and call `close()` when finished. This is optional
and disabled by default. Failed transport requests are not automatically
replayed, since the server may already have executed them.

`ManagedForegroundSession` can start and stop llama-server for a standalone
Python process. Use it as a context manager so the child server is always
closed:

```python
from jetson_pi_foreground import ManagedForegroundSession

with ManagedForegroundSession(
    timeout=300,
) as session:
    action, metadata = session.predict(image_paths, prompt, state, reset=True)
```

Jetson-PI-Edge provides an optional pybind11 module for in-process PI0 and
PI0.5 action inference. The model, multimodal context, backend resources, and
compatible GGML compute graphs remain alive for the lifetime of one
`PIModel` object.

Each `predict()` call is an independent policy tick. It clears request KV
state and position before evaluating the new images, prompt, and robot state.
This does not destroy the model or computation context, so compatible graphs
can still be reused across calls.

## Build

Install pybind11 into the Python environment selected by CMake:

```bash
python3 -m pip install pybind11
```

Configure and build the optional module:

```bash
cmake -S . -B build-python \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DJETSON_PI_BUILD_PYTHON=ON \
  -DLLAMA_BUILD_TOOLS=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF

cmake --build build-python -j --target jetson_pi
```

The module is written to `build-python/bin`. Add that directory to
`PYTHONPATH` or install the configured project:

```bash
PYTHONPATH=build-python/bin python3 -c "import jetson_pi; print(jetson_pi.__doc__)"
```

The Python module is disabled by default and does not add Python or pybind11
to the normal C++ build.
For a CPU-only build, omit `-DGGML_CUDA=ON` and `-DGGML_CUDA_GRAPHS=ON`,
then select `backend="cpu"` without the CUDA runtime flags below.

## Usage

For the recommended **CUDA PI0.5 FP16** configuration, set these variables
before importing/loading the native runtime:

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PI_MODEL"] = "pi05"
os.environ["PI0_PERSISTENT_CUDA_GRAPH"] = "1"
os.environ["PI0_DECODE_UNROLL"] = "10"
os.environ["PI05_FP16_SHARED_CAST"] = "1"
os.environ["PI05_CACHE_TIME_MODULATION"] = "1"
```

Use matching FP16 main/projector GGUFs and `n_threads=8` as in the
[README default example](../README.md#default-in-process-numpy-api).
The time-modulation cache depends only on fixed model weights and the
diffusion schedule, not images, instructions, state or noise. Reuse one model
instance with fixed weights/schedule; new request inputs are still refreshed.
The two new flags are opt-in, and LoRA/binary-debug paths bypass that cache.
For PI0, set `PI_MODEL=pi0`, use its matching model/projector, disable
`PI0_PERSISTENT_CUDA_GRAPH` and omit the `PI05_*` flags. CPU-only inference
should omit these CUDA optimization flags.

```python
import numpy as np
import jetson_pi

model = jetson_pi.load_model(
    model_path="/path/to/pi05-base-F16-10-action.gguf",
    mmproj_path="/path/to/vit.gguf",
    backend="cuda",
    n_views=2,
    image_height=224,
    image_width=224,
    n_threads=8,
)

images = np.empty((2, 224, 224, 3), dtype=np.uint8)
state = np.zeros((32,), dtype=np.float32)

actions = model.predict(images, "pick up the red block", state)
print(actions.shape)       # (action_steps, action_dim)
print(actions.dtype)       # float32

model.close()
```

The model can also be used as a context manager:

```python
with jetson_pi.load_model(
    model_path="/path/to/pi05.gguf",
    mmproj_path="/path/to/vit.gguf",
    backend="cuda",
) as model:
    actions = model.predict(images, "pick up the red block")
```

## Array Contract

`images` must be:

- a NumPy array with dtype `uint8`;
- C-contiguous;
- NHWC with shape `[n_views, image_height, image_width, 3]`.

`state` must be `None` or a C-contiguous one-dimensional NumPy array with
dtype `float32`. A state shorter than the model action dimension is zero
padded by the C API. A state longer than the action dimension is rejected.

The returned action array owns its memory and has dtype `float32` and shape
`[action_steps, action_dim]`.

## Lifetime and Threading

The same `PIModel` object should be reused for a control loop. Creating a new
object reloads the model and loses reusable graph state. Calls on one object
are serialized because the underlying `jetson_pi_pi0` handle is not
thread-safe. Separate objects own separate model contexts.

The Python GIL is released while loading, running inference, and closing the
native model. Calling `predict()` after `close()` raises an error.

Set `LLAMA_GRAPH_REUSE_DISABLE=1` before loading the model only when graph
reuse must be disabled for debugging or comparison.
