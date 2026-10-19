#!/usr/bin/env python3
"""Compare PI0.5 persistent or PI0 unrolled sessions against fresh-server references.

Requires numpy, Pillow, two-image PI0.5 GGUF models, and a fixed 10x32
float32 noise file. Each reference uses a fresh, non-persistent server. The
candidate alternates all cases repeatedly in a single persistent server.
"""

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
from PIL import Image, ImageOps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("server", "model", "mmproj", "noise", "image", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--port", type=int, default=18175)
    parser.add_argument("--gpu", type=int, default=0,
                        help="physical CUDA device index")
    parser.add_argument("--model-kind", choices=("pi0", "pi05"), default="pi05",
                        help="pi0 runs the same regression without the PI0.5-only persistent trajectory")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--cosine-min", type=float, default=0.999)
    parser.add_argument("--max-abs", type=float, default=0.01)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    if args.gpu < 0:
        parser.error("--gpu must be nonnegative")
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", args.port))
    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=False)
    image = str(Path(args.image).resolve())
    inverted = output / "inverted.png"
    with Image.open(image) as source:
        ImageOps.invert(source.convert("RGB")).save(inverted)

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "python"))
    from jetson_pi_foreground import ForegroundSession

    state = np.asarray([1.8731, -1.0370, 1.9652, 7.0876, 0.2546, -9.1432,
                        -0.0147, -0.5037] + [0.0] * 24, dtype=np.float32)
    state_b = state.copy()
    state_b[4] += np.float32(0.5)
    cases = {
        "A": ([image, image], "put both moka pots on the stove", state),
        "B": ([image, image], "move the mug to the other side of the table", state),
        "short": ([image, image], "pick up the cup", state),
        "long": ([image, image], "carefully move the red cup to the right side of the wooden table and release it", state),
        "image": ([image, str(inverted)], "put both moka pots on the stove", state),
        "state": ([image, image], "put both moka pots on the stove", state_b),
    }

    def start(label, persistent):
        env = os.environ.copy()
        for key in ("LLAMA_GRAPH_REUSE_DISABLE", "PI0_DECODE_UNROLL", "PI0_PERSISTENT_CUDA_GRAPH", "PI0_GPU_KV_CACHE"):
            env.pop(key, None)
        env.update(PI_MODEL=args.model_kind, PI0_ACTION_NOISE_BIN=str(Path(args.noise).resolve()),
                   CUDA_VISIBLE_DEVICES=str(args.gpu))
        if persistent:
            env["PI0_PERSISTENT_CUDA_GRAPH"] = "1"
        if label == "persistent":
            env["PI0_DECODE_UNROLL"] = "10"
        command = [str(Path(args.server).resolve()), "-m", str(Path(args.model).resolve()),
                   "--mmproj", str(Path(args.mmproj).resolve()), "-ngl", "99", "--host", "127.0.0.1",
                   "--port", str(args.port), "-np", "1", "-c", "131072", "--chat-template", "vicuna"]
        log = (output / f"{label}-server.log").open("w")
        proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 300
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f"{label} exited with {proc.returncode}; inspect server log")
                try:
                    with urlopen(f"http://127.0.0.1:{args.port}/health", timeout=1) as response:
                        if response.status == 200:
                            break
                except (URLError, TimeoutError):
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError(f"{label} did not become ready")
                time.sleep(0.5)
        except BaseException:
            stop(proc, log)
            raise
        return proc, log, ForegroundSession(f"http://127.0.0.1:{args.port}", timeout=300)

    def stop(proc, log):
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        log.close()

    def predict(session, case):
        images, instruction, current_state = case
        action, _ = session.predict(images, instruction, current_state, reset=True)
        return np.asarray(action, dtype=np.float64)

    references = {}
    for label, case in cases.items():
        print(f"fresh reference: {label}", flush=True)
        proc, log, session = start(f"reference-{label}", False)
        try:
            reference = predict(session, case)
        finally:
            stop(proc, log)
        if reference.shape != (10, 32) or not np.isfinite(reference).all():
            raise RuntimeError(f"invalid fresh reference {label}: shape={reference.shape}")
        references[label] = reference
    np.savez(output / "references.npz", **references)

    proc, log, session = start("persistent", args.model_kind == "pi05")
    comparisons = []
    actions = {}
    # Revisit A after every other case and alternate sequence direction on
    # successive rounds to exercise both growing and shrinking token counts.
    try:
        for round_index in range(args.rounds):
            order = ["B", "short", "long", "image", "state"]
            if round_index % 2:
                order.reverse()
            for label in ["A"] + [item for name in order for item in (name, "A")]:
                candidate = predict(session, cases[label])
                reference = references[label]
                finite = candidate.shape == reference.shape and bool(np.isfinite(candidate).all())
                if finite:
                    denom = np.linalg.norm(candidate) * np.linalg.norm(reference)
                    cosine = float(np.sum(candidate * reference) / denom) if denom else None
                    maximum = float(np.max(np.abs(candidate - reference)))
                    mean = float(np.mean(np.abs(candidate - reference)))
                else:
                    cosine = maximum = mean = None
                passed = finite and cosine is not None and cosine > args.cosine_min and maximum <= args.max_abs
                record = {"round": round_index + 1, "case": label, "finite": finite,
                          "cosine": cosine, "max_abs": maximum, "mean_abs": mean, "passed": bool(passed)}
                comparisons.append(record)
                actions[f"round{round_index + 1}_{len(comparisons)}_{label}"] = candidate
                print(json.dumps(record, allow_nan=False), flush=True)
    finally:
        stop(proc, log)
    np.savez(output / "persistent-actions.npz", **actions)
    summary = {"model_kind": args.model_kind, "persistent_enabled": args.model_kind == "pi05",
               "candidate_unroll": 10, "gpu": args.gpu,
               "rounds": args.rounds, "cosine_min": args.cosine_min, "max_abs_limit": args.max_abs,
               "passed": all(item["passed"] for item in comparisons), "comparisons": comparisons}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    if not summary["passed"]:
        raise SystemExit("persistent-session regression failed; see summary.json")


if __name__ == "__main__":
    main()
