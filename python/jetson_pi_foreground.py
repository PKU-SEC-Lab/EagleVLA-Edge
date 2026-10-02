"""NumPy client for the llama-server foreground session API."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

DEFAULT_SERVER_PATH = "/home/yzb/Jetson-PI-2-graph-kernel/build-graph/bin/llama-server"
DEFAULT_MODEL_PATH = "/data/home/yzb/model/gguf/models--lerobot--pi0_base/snapshots/25c379b52ba2ff8788cab921758a3cc3fe3f77f2/pi0_base-F16_new.gguf"
DEFAULT_MMPROJ_PATH = "/data/home/yzb/model/gguf/vit/mmproj-model-f16_NEW.gguf"


class ForegroundSession:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        timeout: float = 300.0,
        compact: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.compact = compact
        self._state_dim = 0

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = Request(
            self.base_url + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"foreground request failed ({error.code}): {detail}") from error
        except URLError as error:
            raise RuntimeError(f"could not connect to foreground server at {self.base_url}: {error}") from error
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(f"foreground server error: {result['error']}")
        return result

    def reset(self) -> dict:
        return self._request("POST", "/foreground/reset", {})

    def load_image(self, path: str) -> dict:
        if not path:
            raise ValueError("image path must not be empty")
        return self._request("POST", "/foreground/image", {"path": path})

    def set_state(self, state: np.ndarray) -> dict:
        state = np.asarray(state)
        if state.dtype != np.float32 or state.ndim != 1:
            raise ValueError("state must be a one-dimensional float32 NumPy array")
        self._state_dim = int(state.size)
        value = ",".join(format(float(item), ".9g") for item in state)
        return self._request("PUT", "/foreground/state", {"state": value})

    def infer(self, prompt: str) -> tuple[np.ndarray, dict]:
        if not prompt:
            raise ValueError("prompt must not be empty")
        response = self._request(
            "POST",
            "/foreground/infer",
            {"text": prompt, "compact": self.compact},
        )
        action = np.asarray(response.get("action_final_raw", response.get("action_final")), dtype=np.float32)
        if action.ndim == 1:
            steps = int(response.get("action_steps", 0))
            dim = int(response.get("action_dim", 0))
            if dim <= 0 and self._state_dim > 0 and action.size % self._state_dim == 0:
                dim = self._state_dim
                steps = action.size // dim
            if steps > 0 and dim > 0 and action.size == steps * dim:
                action = action.reshape(steps, dim)
        if action.ndim != 2:
            raise RuntimeError(f"foreground server returned an invalid action shape: {action.shape}")
        return np.ascontiguousarray(action), response

    def predict(self, image_paths, prompt: str, state: np.ndarray, reset: bool = False) -> tuple[np.ndarray, dict]:
        if reset:
            self.reset()
        for path in image_paths:
            self.load_image(str(path))
        self.set_state(state)
        return self.infer(prompt)


class ManagedForegroundSession(ForegroundSession):
    def __init__(
        self,
        server_path: str = DEFAULT_SERVER_PATH,
        model_path: str = DEFAULT_MODEL_PATH,
        mmproj_path: str = DEFAULT_MMPROJ_PATH,
        host: str = "127.0.0.1",
        port: int = 8080,
        gpu: int = 0,
        n_gpu_layers: int = 37,
        parallel: int = 1,
        chat_template: str = "vicuna",
        noise_path: str | None = None,
        startup_timeout: float = 300.0,
        timeout: float = 300.0,
        compact: bool = True,
        log_path: str | None = None,
    ):
        super().__init__(f"http://{host}:{port}", timeout, compact=compact)
        for name, path in (
            ("server", server_path),
            ("model", model_path),
            ("mmproj", mmproj_path),
        ):
            if not Path(path).is_file():
                raise FileNotFoundError(f"{name} file not found: {path}")

        command = [
            server_path,
            "-m", model_path,
            "--mmproj", mmproj_path,
            "-ngl", str(n_gpu_layers),
            "--chat-template", chat_template,
            "--host", host,
            "--port", str(port),
            "-np", str(parallel),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env.setdefault("PI_MODEL", "pi0")
        if noise_path is not None:
            if not Path(noise_path).is_file():
                raise FileNotFoundError(f"noise file not found: {noise_path}")
            env["PI0_ACTION_NOISE_BIN"] = noise_path

        self._log_file = open(log_path, "a") if log_path is not None else None
        output = self._log_file if self._log_file is not None else subprocess.DEVNULL
        self._process = subprocess.Popen(command, env=env, stdout=output, stderr=output)
        try:
            self._wait_until_ready(startup_timeout)
        except Exception:
            self.close()
            raise

    def _wait_until_ready(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError(f"llama-server exited during startup with code {self._process.returncode}")
            try:
                self._request("GET", "/health")
                return
            except RuntimeError:
                time.sleep(0.2)
        raise TimeoutError(f"llama-server did not become ready within {timeout:g} seconds")

    def close(self) -> None:
        process = getattr(self, "_process", None)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        self._process = None
        log_file = getattr(self, "_log_file", None)
        if log_file is not None:
            log_file.close()
            self._log_file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
