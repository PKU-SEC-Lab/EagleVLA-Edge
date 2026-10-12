#!/usr/bin/env python3
"""Build, validate Gemma RMSNorm fusion, benchmark, then collect warm CUDA trace.

Only the matching yzb test server on localhost:18081 may be replaced. This
runner does not claim completion of the remaining FP8/GEMM/attention ports.
"""
import hashlib
import json
import math
import os
import re
import runpy
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import types
import urllib.request
from pathlib import Path


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def command_run(args, path, cwd, env, timeout=900):
    print("运行:", " ".join(map(str, args)), flush=True)
    started = time.monotonic()
    with path.open("w") as log:
        p = subprocess.Popen(list(map(str, args)), cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                             stdout=log, stderr=subprocess.STDOUT)
        while True:
            try:
                code = p.wait(timeout=10)
                break
            except subprocess.TimeoutExpired:
                print(f"处理中：{time.monotonic()-started:.0f}s；日志 {path.name}", flush=True)
                if time.monotonic() - started > timeout:
                    p.terminate()
                    p.wait(timeout=15)
                    raise RuntimeError(f"命令超过 {timeout}s；见 {path}")
    text = path.read_text(errors="replace")
    need(code == 0, f"命令失败 ({code})：\n{text[-5000:]}")
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def percentile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def shadow_records(text):
    records = []
    for line in text.splitlines():
        if "flashrt_rmsnorm_shadow:" in line:
            record = dict(re.findall(r"(\w+)=([^\s]+)", line))
            need(record.get("math") == "ggml" and record.get("propagate") == "ggml", "shadow 模式不符")
            need(int(record["nonfinite"]) == 0, "RMSNorm shadow 出现非有限值")
            for key in ("ab_max", "ab_rmse", "ab_rel_l2"):
                need(float(record[key]) == 0, "RMSNorm shadow 未逐节点一致：" + line)
            records.append(record)
    need({1024, 2048} <= {int(r["dim"]) for r in records}, "未同时命中 decoder/encoder RMSNorm")
    return records


def gguf_inventory(path):
    """Read tensor headers only; no NumPy, mmap or model data allocation."""
    size = path.stat().st_size
    with path.open("rb") as f:
        def raw(n):
            need(0 <= n <= size - f.tell(), "GGUF 头被截断")
            value = f.read(n)
            need(len(value) == n, "GGUF 读取失败")
            return value
        def number(fmt):
            return struct.unpack("<" + fmt, raw(struct.calcsize(fmt)))[0]
        def skip(n):
            need(0 <= n <= size - f.tell(), "GGUF 字段越界")
            f.seek(n, 1)
        def string():
            n = number("Q")
            need(n <= 16 * 1024 * 1024, "GGUF 字符串字段异常")
            return raw(n).decode("utf-8")
        fixed = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        def value(kind, depth=0):
            need(depth < 8, "GGUF metadata 嵌套异常")
            if kind in fixed:
                skip(fixed[kind])
            elif kind == 8:
                skip(number("Q"))
            elif kind == 9:
                element, count = number("I"), number("Q")
                need(count <= size, "GGUF 数组长度异常")
                if element in fixed:
                    skip(count * fixed[element])
                else:
                    for _ in range(count):
                        value(element, depth + 1)
            else:
                raise RuntimeError(f"未知 GGUF metadata 类型 {kind}")
        need(raw(4) == b"GGUF", "文件不是 little-endian GGUF")
        version = number("I")
        need(version in (2, 3), f"暂不支持 GGUF version {version}")
        count, metadata = number("Q"), number("Q")
        need(count <= 1000000 and metadata <= 1000000, "GGUF 头部计数异常")
        for _ in range(metadata):
            string()
            value(number("I"))
        tensors, by_type = [], {}
        type_names = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
                      8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
                      13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 30: "BF16"}
        for _ in range(count):
            name, dims = string(), number("I")
            need(1 <= dims <= 4, "GGUF tensor 维数异常")
            shape = [number("Q") for _ in range(dims)]
            need(all(d > 0 for d in shape), "GGUF tensor 维度异常")
            kind, offset = number("I"), number("Q")
            dtype = type_names.get(kind, f"GGML_TYPE_{kind}")
            by_type[dtype] = by_type.get(dtype, 0) + 1
            tensors.append({"name": name, "ggml_type": kind, "dtype": dtype, "ne": shape, "offset": offset})
        return {"path": str(path), "file_size": size, "version": version, "tensor_count": count,
                "dtype_counts": by_type, "tensors": tensors}


def main():
    root = Path.home() / "Jetson-PI-Edge-flashrt-sm89"
    h = types.SimpleNamespace(**runpy.run_path(str(root / "run-ggml-math-shadow.py")))
    profile_mod = types.SimpleNamespace(**runpy.run_path(str(Path(__file__).with_name("run-sm89-profile.py"))))
    proof_path = h.RESULTS / "ggml-math-terminal-9f94pkb6/summary.json"
    proof = json.loads(proof_path.read_text())
    need(proof["status"] == "equal_in_both_settings", "此前 LayerNorm 验证未完成")
    need(hashlib.sha256(h.IMAGE.read_bytes()).hexdigest() == proof["input"]["image_sha256"], "固定图片发生变化")
    binary = (root / "build-flashrt-sm89/bin/llama-server").resolve(strict=True)
    model, mmproj = h.MODEL_DIR / "pi_llm.gguf", h.MODEL_DIR / "mmproj-model-f16.gguf"
    for path in (model, mmproj, root / "ggml/src/ggml-cuda/flashrt-rmsnorm.cu"):
        need(path.is_file(), f"缺少文件 {path}")
    cmd = list(proof["command"])
    need(cmd[0] == str(binary), "二进制与此前验证不符")
    reference = json.loads(h.BASELINE.read_text())["records"][0]["response"]
    keys = ("action", "action_final_raw", "action_final")
    for key in keys:
        h.flat_action(reference, key)
    out = Path(tempfile.mkdtemp(prefix="rmsnorm-sprint-", dir=h.RESULTS))
    report = {"status": "running", "scope": "GGML-math RMSNorm fusion validation and warm CUDA profile",
              "full_port_complete": False, "prior_proof": str(proof_path), "stages": {}, "runs": []}
    print("本次结果目录:", out, flush=True)
    cases = [
        {"id": "fixed", "text": h.PROMPT, "state": "0," * 31 + "0"},
        {"id": "changed-text", "text": "move the object to the left", "state": "0," * 31 + "0"},
        {"id": "changed-state", "text": h.PROMPT, "state": ",".join(map(str, [.1, -.1] + [0] * 30))},
    ]
    report["inputs"] = {"image": str(h.IMAGE), "image_sha256": proof["input"]["image_sha256"], "cases": cases,
                        "scope": "one image, three prompt/state combinations; not task accuracy or FP8 calibration"}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    metric_keys = ("vit_ms", "encode_ms", "decode_ms", "total_ms", "handler_total_ms", "infer_http_ms")
    targets = {}

    def env_for(mode, graphs=True, shadow=False, trace=False):
        env = dict(os.environ, **h.MODE_ENV)
        env.update(GGML_CUDA_FLASHRT=str(mode), GGML_CUDA_FLASHRT_GGML_MATH="1",
                   GGML_CUDA_FLASHRT_RMSNORM="1", GGML_CUDA_FLASHRT_SHADOW=str(int(shadow)),
                   GGML_CUDA_FLASHRT_TRACE=str(int(trace)), GGML_CUDA_DISABLE_FUSION="0",
                   PI_MODEL_DEBUG="0", LLAMA_PI0_PERF="0")
        for key in ("PI0_ACTION_NOISE_BIN", "PI0_NORM_STATS_JSON", "PI_MODEL_DEBUG_DUMP_FILE", "PI05_DEBUG_DUMP_FILE",
                    "GGML_CUDA_DISABLE_GRAPHS"):
            env.pop(key, None)
        if not graphs:
            env["GGML_CUDA_DISABLE_GRAPHS"] = "1"
        return env

    def compare(response, expected):
        for key in keys:
            d = h.difference(response, expected, key)
            need(d["max_abs"] == 0, f"{key}: max_abs={d['max_abs']:.9g}, RMSE={d['rmse']:.9g}")

    def infer(case=0, expected=None):
        data = cases[case]
        need(h.request(opener, "POST", "/foreground/reset").get("reset") is True, "reset 失败")
        need(h.request(opener, "POST", "/foreground/image", {"path": str(h.IMAGE)}, timeout=30).get("loaded") is True,
             "图片加载失败")
        state = h.request(opener, "PUT", "/foreground/state", {"state": data["state"]})
        need(state.get("state_dim") == 32 and state.get("loaded") is True, "state 加载失败")
        start = time.perf_counter()
        response = h.request(opener, "POST", "/foreground/infer", {"text": data["text"]}, timeout=120)
        record = {"case": data["id"], "response": response, "infer_http_ms": (time.perf_counter() - start) * 1000}
        try:
            for key in keys:
                h.flat_action(response, key)
            if expected is not None:
                compare(response, expected)
        except BaseException:
            # Preserve nonfinite evidence as strings in valid JSON; a strict
            # JSON writer must not hide the original numerical failure.
            evidence = json.loads(json.dumps({"actual": record, "expected": expected}),
                                  parse_constant=lambda value: value)
            h.write_json(out / "failed-response.json", evidence)
            raise
        return record

    def run_server(label, mode, graphs, work, *, shadow=False, trace=False, verbosity=5):
        need(h.listener_pid() is None, "18081 被占用，未启动新服务")
        env = env_for(mode, graphs, shadow, trace)
        args = list(cmd)
        args[args.index("-lv") + 1] = str(verbosity)
        log_path = out / (label + ".log")
        stage = {"mode": mode, "graphs_allowed": graphs, "shadow": shadow,
                 "env": {k: v for k, v in env.items() if k.startswith(("GGML_", "PI_MODEL", "CUDA_VISIBLE"))},
                 "records": []}
        report["stages"][label] = stage
        print("\n阶段:", label, flush=True)
        with log_path.open("w") as log:
            child = subprocess.Popen(args, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                h.wait_ready(child, opener)
                work(stage["records"])
                need(child.poll() is None and h.listener_pid() == child.pid, "服务进程发生变化")
                stage["status"] = "requests_passed"
            finally:
                h.stop_child(child)
                h.write_json(out / (label + ".json"), stage)
        h.write_json(out / "summary.json", report)
        return log_path.read_text(errors="replace")

    try:
        need(Path('/proc/self/stat').read_text().split(' ', 1)[0] == str(os.getpid()),
             "procfs 与进程 PID 不在同一命名空间；请在 gpu4 的普通 SSH 终端执行")
        base_env = env_for(0)
        gpu = command_run(["nvidia-smi", "-i", "2", "--query-gpu=index,name,driver_version,memory.used",
                           "--format=csv"], out / "gpu-before.csv", root, base_env, timeout=20)
        need("RTX 4090" in gpu, "GPU 2 不是预期的 RTX 4090")
        print(gpu, flush=True)
        report["nsys"] = profile_mod.find_nsys()
        inventory = [gguf_inventory(path) for path in (model, mmproj)]
        h.write_json(out / "model-tensors.json", inventory)
        report["model_tensors"] = str(out / "model-tensors.json")
        for item in inventory:
            print(Path(item["path"]).name, "tensor dtype:", item["dtype_counts"], flush=True)
        print(command_run([sys.executable, root / "tests/test-sm89-sprint-runner.py"],
                          out / "runner-tests.log", root, base_env, timeout=90), flush=True)
        h.stop_old_server(binary, model, mmproj)
        cmake = shutil.which("cmake")
        need(cmake is not None, "PATH 未找到 cmake")
        shutil.copyfile(root / "build-flashrt-sm89/CMakeCache.txt", out / "CMakeCache-before.txt")
        command_run([cmake, "-S", root, "-B", root / "build-flashrt-sm89", "-DGGML_CUDA=ON",
                     "-DGGML_CUDA_FLASHRT=ON", "-DCMAKE_CUDA_ARCHITECTURES=89", "-DLLAMA_BUILD_TESTS=ON"],
                    out / "configure.log", root, base_env)
        command_run([cmake, "--build", root / "build-flashrt-sm89", "--target", "llama-server",
                     "test-backend-ops", "test-flashrt-rmsnorm-guard", "-j", "8"], out / "build.log", root, base_env)
        bindir = binary.parent
        print(command_run([bindir / "test-flashrt-rmsnorm-guard"], out / "guard.log", root, base_env), flush=True)
        for mode in (0, 1):
            text = command_run([bindir / "test-backend-ops", "test", "-b", "CUDA0", "-o", "FLASHRT_RMSNORM"],
                               out / f"ops-{mode}.log", root, env_for(mode, trace=True), timeout=180)
            need("RTX 4090" in text and "25/25 tests passed" in text, "GPU 算子测试未完整执行")
            need(("flashrt_rmsnorm: enqueued" in text) == bool(mode), "算子测试实际命中不符")
            print(f"RMSNorm CUDA0 mode={mode}: 25/25，通过", flush=True)
        log = run_server("shadow", 1, False,
                         lambda records: records.append(infer(expected=reference)), shadow=True, verbosity=4)
        records = shadow_records(log)
        report["rmsnorm_shadow"] = {"count": len(records), "records": records}
        print(f"RMSNorm shadow: {len(records)} 条，逐节点差异为 0", flush=True)
        for graphs in (False, True):
            sequence = [0] * 6 + [1] * 6 + [2] * 6 + [0] * 4 if graphs else [0, 1, 2]
            for mode in (0, 1):
                label = f"e2e-graphs-{int(graphs)}-mode-{mode}"
                def work(saved):
                    for index, case in enumerate(sequence):
                        expected = targets.get((graphs, case))
                        if case == 0:
                            expected = reference
                        record = infer(case, expected)
                        if mode == 0:
                            targets[(graphs, case)] = record["response"]
                        else:
                            compare(record["response"], targets[(graphs, case)])
                        saved.append(record)
                        print(f"{label}: {index+1}/{len(sequence)}，{cases[case]['id']} 动作一致", flush=True)
                text = run_server(label, mode, graphs, work, trace=True)
                matches = re.findall(r"flashrt_rmsnorm: enqueued rows=(\d+) dim=(\d+) node=(\S+) math=ggml", text)
                need((len(matches) > 0) == bool(mode), "模型融合实际命中不符")
                if mode:
                    need({1024, 2048} <= {int(m[1]) for m in matches}, "模型命中未覆盖 encoder/decoder")
                warmups = text.count("CUDA graph warmup complete")
                need((warmups > 0) == graphs, "CUDA Graph 设置或预热证据不符")
                report["stages"][label].update(rmsnorm_enqueues=len(matches), graph_warmups=warmups)
        report["correctness"] = "equal_for_three_input_cases_in_both_graph_settings"
        for index, mode in enumerate((0, 1, 1, 0), 1):
            def measure(saved):
                for i in range(5):
                    infer(expected=reference)
                    print(f"ABBA {index}/4 warmup {i+1}/5", flush=True)
                for i in range(20):
                    record = infer(expected=reference)
                    saved.append(record)
                    print(f"ABBA {index}/4 measure {i+1}/20: {record['infer_http_ms']:.3f} ms", flush=True)
            label = f"perf-{index}-mode-{mode}"
            log = run_server(label, mode, True, measure, verbosity=3)
            need("_shadow:" not in log and "flashrt_rmsnorm: enqueued" not in log, "性能阶段仍有诊断输出")
            samples = []
            for record in report["stages"][label]["records"]:
                response = record["response"]
                sample = {k: float(response[k]) for k in metric_keys[:4]}
                sample["handler_total_ms"] = float(response["timing_breakdown_ms"]["handler_total_ms"])
                sample["infer_http_ms"] = record["infer_http_ms"]
                need(all(math.isfinite(v) and v > 0 for v in sample.values()), "时延字段异常")
                samples.append(sample)
            report["runs"].append({"mode": mode, "samples": samples})
        print("\nmetric,A_p50_ms,B_p50_ms,A_p95_ms,B_p95_ms,p50_latency_reduction_pct", flush=True)
        report["performance"] = {"A": "original GGML", "B": "GGML-math LayerNorm + Gemma RMSNorm",
                                 "order": [0, 1, 1, 0], "metrics": {}}
        for key in metric_keys:
            a = [s[key] for r in report["runs"] if r["mode"] == 0 for s in r["samples"]]
            b = [s[key] for r in report["runs"] if r["mode"] == 1 for s in r["samples"]]
            ap, bp = percentile(a, .5), percentile(b, .5)
            aq, bq, reduction = percentile(a, .95), percentile(b, .95), 100 * (ap-bp) / ap
            report["performance"]["metrics"][key] = {"A_p50": ap, "B_p50": bp, "A_p95": aq,
                                                        "B_p95": bq, "p50_latency_reduction_pct": reduction}
            print(f"{key},{ap:.3f},{bp:.3f},{aq:.3f},{bq:.3f},{reduction:.2f}")
        print("A=原 GGML；B=GGML-math LayerNorm + RMSNorm，属于组合收益。", flush=True)
        report["rmsnorm_status"] = "validated_for_tested_inputs_and_benchmarked"
        args = list(cmd)
        args[args.index("-lv") + 1] = "3"
        report["profile"] = profile_mod.profile(h, args, env_for(0), opener,
                                               lambda: infer(expected=reference), out / "profile-baseline")
        report["status"] = "rmsnorm_stage_complete" if report["profile"]["status"] == "captured" else "profile_pending"
    except BaseException as exc:
        report.update(status="incomplete", error=f"{type(exc).__name__}: {exc}")
        print("阶段未完成:", report["error"], flush=True)
        raise
    finally:
        h.write_json(out / "summary.json", report)
        print("\nstatus:", report["status"], flush=True)
        print("correctness:", report.get("correctness", "未完成"), flush=True)
        print("RMSNorm shadow:", report.get("rmsnorm_shadow", {}).get("count", 0), "条", flush=True)
        print("\n已保存:", out / "summary.json", flush=True)
        print("完整 kernel 移植仍需完成覆盖表中的其余区域；本次结果只报告实际验收阶段。", flush=True)


if __name__ == "__main__":
    main()
