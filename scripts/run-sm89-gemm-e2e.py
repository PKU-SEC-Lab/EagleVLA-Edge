#!/usr/bin/env python3
"""Verify mode 2 GEMM output propagation and, only after success, time ABBA.

Mode 0 keeps the previously accepted GGML-math LayerNorm and RMSNorm enabled.
Mode 1 (cuBLASLt) is never used by this model runner. Existing listeners are
never stopped. Only a Popen child owned by this invocation is cleaned up.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
import tempfile
import time
import types
import urllib.request


SHAPES = {(1024, 2048, 11), (1024, 256, 11), (2048, 1024, 11),
          (1024, 4096, 11), (4096, 1024, 11)}
ACTION_KEYS = ("action", "action_final_raw", "action_final")
METRICS = ("vit_ms", "encode_ms", "decode_ms", "total_ms",
           "handler_total_ms", "infer_http_ms")


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_json(path, value):
    path = Path(path)
    pending = path.with_name(path.name + ".tmp")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                  allow_nan=False) + "\n")
    pending.replace(path)


def evidence_json(value):
    """Keep nonfinite response evidence without emitting invalid JSON."""
    return json.loads(json.dumps(value), parse_constant=lambda token: token)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values, q):
    values = sorted(values)
    need(bool(values), "时延样本为空")
    position = (len(values) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def verify_micro(root, path):
    micro = json.loads(path.read_text())
    need(micro.get("status") == "microtests_complete", "GEMM micro 阶段尚未成功；未启动模型服务")
    gate = micro.get("mode2_correctness", {})
    need(gate.get("status") == "passed" and gate.get("binary_equal") is True and
         gate.get("replay_verified") is True and
         gate.get("minimum_actual_graph_launches_per_case", 0) >= 6,
         "缺少 mode 2 exact / 实际 graph launch 通过记录")
    actual_shapes = {(r["k"], r["n"], r["m"]) for r in gate.get("shapes", [])}
    need(actual_shapes == SHAPES and set(gate.get("cases", [])) ==
         {"normal", "cancellation", "bounded"}, "micro 覆盖未包含全部 5 种形状和 3 种数值用例")
    manifest_path = Path(micro["build"]["manifest"]).resolve(strict=True)
    need(micro["build"].get("manifest_sha256") == sha256(manifest_path),
         "micro 验收时的构建清单与当前文件不一致，或缺少冻结指纹；请重新运行 micro")
    manifest = json.loads(manifest_path.read_text())
    build = Path(micro["build"]["directory"]).resolve(strict=True)
    need(micro.get("build_directory") == str(build), "micro 的构建目录字段不一致")
    module = types.SimpleNamespace(**runpy.run_path(str(root / "scripts/run-sm89-gemm-stage.py")))
    need(manifest.get("root") == str(root), "micro 验收的仓库路径不匹配")
    need(manifest.get("sources") == module.source_hashes(root), "micro 验收后源码已改变；请重新构建并运行 micro")
    need(manifest.get("artifacts") == module.artifact_hashes(build),
         "micro 验收后构建产物已改变；请重新运行 micro")
    need(manifest.get("cache_sha256") == sha256(build / "CMakeCache.txt"),
         "micro 验收后 CMake 配置已改变")
    need("bin/llama-server" in manifest["artifacts"], "micro 清单未包含 llama-server；不能验收另一份 server")
    need(micro.get("gpu", {}).get("compute_capability") == "8.9" and
         isinstance(micro["gpu"].get("uuid"), str) and micro["gpu"]["uuid"].startswith("GPU-"),
         "micro 记录缺少有效的 SM89 GPU UUID")
    return micro, build, manifest, module


def verify_model_trace(text, mode, graphs):
    need("_shadow:" not in text, "E2E 中出现 shadow 记录；未验收真实输出传播")
    lines = [line for line in text.splitlines() if "flashrt_gemm:" in line]
    matched, captured, nodes = set(), set(), set()
    for line in lines:
        fields = dict(re.findall(r"(\w+)=([^\s]+)", line))
        need(mode == 2 and fields.get("mode") == "ggml_mma16", "模型 GEMM 实际 dispatch 与请求不符：" + line)
        shape = tuple(int(fields[k]) for k in ("K", "N", "M"))
        need(shape in SHAPES, "模型 GEMM 命中未验收的形状：" + line)
        need(fields.get("capture") in ("0", "1"), "GEMM TRACE 缺少有效的 capture 状态：" + line)
        need(graphs or fields["capture"] == "0", "Graphs off 阶段出现正在 capture 的 GEMM：" + line)
        need(bool(fields.get("node")) and bool(fields.get("weight")), "GEMM TRACE 缺少节点/权重身份")
        matched.add(shape)
        if fields["capture"] == "1":
            captured.add(shape)
        nodes.add((fields["node"], fields["weight"]))
    if mode == 2:
        need(matched == SHAPES, f"模型 GEMM 未命中全部 5 种形状；已命中 {sorted(matched)}")
        if graphs:
            need(captured == SHAPES, f"模型 graph capture 未覆盖全部 5 种候选 GEMM；已观察 {sorted(captured)}")
    else:
        need(not lines, "GEMM=0 对照出现候选 dispatch")
    warmups = text.count("CUDA graph warmup complete")
    need((warmups > 0) == graphs, "模型 CUDA Graph 设置与 warmup 日志不符")
    return {"gemm_trace_count": len(lines), "gemm_node_weight_pairs": len(nodes),
            "shapes": [{"k": k, "n": n, "m": m} for k, n, m in sorted(matched)],
            "shapes_observed_during_capture": [{"k": k, "n": n, "m": m} for k, n, m in sorted(captured)],
            "graph_warmup_complete_observed": warmups,
            "model_graph_replay_count": None,
            "graph_evidence_scope": "candidate dispatch capture state and backend warmup logs; model replay count is not independently instrumented"}


def file_identity(path):
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "device": stat.st_dev, "inode": stat.st_ino}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--micro-summary", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--skip-performance", action="store_true",
                        help="完成全部模型数值检查后停止，不运行 ABBA 性能比较")
    args = parser.parse_args(argv)
    root = args.root.resolve(strict=True)
    results = root / "results/flashrt-sm89"
    results.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="gemm-mma16-e2e-", dir=results))
    report = {"schema_version": 1, "status": "running", "full_port_complete": False,
              "scope": "five SM89 M=11 decoder GEMM shapes; one image and three prompt/state cases",
              "micro_summary": str(args.micro_summary.resolve()), "stages": {}, "runs": [],
              "runner_sha256": sha256(Path(__file__)), "cleanup_errors": []}
    print("本次结果目录:", out, flush=True)
    stage_number = 0
    try:
        need(Path("/proc/self/stat").read_text().split(" ", 1)[0] == str(os.getpid()),
             "procfs 与当前 PID 不在同一命名空间；请在 gpu4 普通 SSH 终端执行")
        micro, build, manifest, micro_mod = verify_micro(root, args.micro_summary.resolve(strict=True))
        report["build_manifest"] = manifest
        report["micro_mode2_correctness"] = micro["mode2_correctness"]
        helper_path = root / "run-ggml-math-shadow.py"
        h = types.SimpleNamespace(**runpy.run_path(str(helper_path)))
        report["helper_sha256"] = sha256(helper_path)
        if hasattr(h, "ROOT"):
            need(Path(h.ROOT).resolve() == root, "已有服务 helper 指向另一份仓库")
        need(h.listener_pid() is None, "18081 已被占用；未停止任何已有进程。请先退出自己的测试服务")
        report["gpu"] = micro_mod.gpu_snapshot(micro["gpu"]["uuid"], out)
        need(report["gpu"]["uuid"] == micro["gpu"]["uuid"], "E2E GPU 与 micro 验收不一致")
        binary = (build / "bin/llama-server").resolve(strict=True)
        model = (Path(h.MODEL_DIR) / "pi_llm.gguf").resolve(strict=True)
        mmproj = (Path(h.MODEL_DIR) / "mmproj-model-f16.gguf").resolve(strict=True)
        image = Path(h.IMAGE).resolve(strict=True)
        report["models"] = [file_identity(model), file_identity(mmproj)]
        cases = [
            {"id": "fixed", "text": h.PROMPT, "state": "0," * 31 + "0"},
            {"id": "changed-text", "text": "move the object to the left", "state": "0," * 31 + "0"},
            {"id": "changed-state", "text": h.PROMPT,
             "state": ",".join(map(str, [.1, -.1] + [0] * 30))},
        ]
        report["input"] = {"image": str(image), "image_sha256": sha256(image), "cases": cases}
        report["action_comparison"] = {"keys": list(ACTION_KEYS), "initial_noise_field": "action",
                                      "acceptance": "finite, 10x32, exact numerical equality"}
        command = [str(binary), "-m", str(model), "--mmproj", str(mmproj), "-ngl", "99",
                   "-c", "2048", "--parallel", "1", "--flash-attn", "on", "--no-warmup",
                   "--host", "127.0.0.1", "--port", "18081", "-lv", "5"]
        report["command"] = command
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        targets = {}

        def env_for(mode, graphs, trace):
            env = dict(os.environ, **h.MODE_ENV)
            for key in list(env):
                if key.startswith("GGML_CUDA_FLASHRT"):
                    env.pop(key)
            for key in ("PI0_ACTION_NOISE_BIN", "PI0_NORM_STATS_JSON", "PI_MODEL_DEBUG_DUMP_FILE",
                        "PI05_DEBUG_DUMP_FILE", "GGML_CUDA_DISABLE_GRAPHS"):
                env.pop(key, None)
            env.update(CUDA_VISIBLE_DEVICES=micro["gpu"]["uuid"], PI_MODEL="pi0",
                       GGML_CUDA_FLASHRT="1", GGML_CUDA_FLASHRT_GGML_MATH="1",
                       GGML_CUDA_FLASHRT_RMSNORM="1", GGML_CUDA_FLASHRT_GEMM=str(mode),
                       GGML_CUDA_FLASHRT_SHADOW="0", GGML_CUDA_FLASHRT_GEMM_SHADOW="0",
                       GGML_CUDA_FLASHRT_GEMM_GRAPH_TRACE="0",
                       GGML_CUDA_FLASHRT_TRACE=str(int(trace)), GGML_CUDA_DISABLE_FUSION="0",
                       PI_MODEL_DEBUG="0", LLAMA_PI0_PERF="0")
            if not graphs:
                env["GGML_CUDA_DISABLE_GRAPHS"] = "1"
            return env

        def compare(response, expected):
            for key in ACTION_KEYS:
                h.flat_action(response, key)
                h.flat_action(expected, key)
                diff = h.difference(response, expected, key)
                need(diff["max_abs"] == 0 and diff["rmse"] == 0,
                     f"{key}: max_abs={diff['max_abs']:.9g}, RMSE={diff['rmse']:.9g}")

        def infer(child, stage, case_index, expected, phase, request_index):
            data = cases[case_index]
            need(child.poll() is None and h.listener_pid() == child.pid, "请求前服务进程身份发生变化")
            need(h.request(opener, "POST", "/foreground/reset").get("reset") is True, "reset 失败")
            need(h.request(opener, "POST", "/foreground/image", {"path": str(image)}, timeout=30).get("loaded") is True,
                 "图片加载失败")
            state = h.request(opener, "PUT", "/foreground/state", {"state": data["state"]})
            need(state.get("loaded") is True and state.get("state_dim") == 32, "state 加载失败")
            start = time.perf_counter()
            response = h.request(opener, "POST", "/foreground/infer", {"text": data["text"]}, timeout=120)
            elapsed = (time.perf_counter() - start) * 1000
            record = {"case": data["id"], "phase": phase, "request_index": request_index,
                      "infer_http_ms": elapsed, "response": response}
            try:
                need(child.poll() is None and h.listener_pid() == child.pid, "请求后服务进程身份发生变化")
                for key in ACTION_KEYS:
                    h.flat_action(response, key)
                if expected is not None:
                    compare(response, expected)
            except BaseException:
                path = out / (stage["label"] + f"-failed-response-{request_index}.json")
                write_json(path, evidence_json({"actual": record, "expected": expected}))
                stage["failed_response"] = str(path)
                raise
            return record

        def run_server(label, mode, graphs, work, trace=True):
            nonlocal stage_number
            stage_number += 1
            need(h.listener_pid() is None, "18081 已被占用；未停止任何已有进程")
            need(report["models"] == [file_identity(model), file_identity(mmproj)] and
                 report["input"]["image_sha256"] == sha256(image), "测试过程中模型或图片发生变化")
            args_for_server = list(command)
            args_for_server[-1] = "5" if trace else "3"
            env = env_for(mode, graphs, trace)
            log_path = out / (label + ".log")
            stage = {"label": label, "mode": mode, "graphs_allowed": graphs, "trace": trace,
                     "status": "starting", "records": [], "log": str(log_path),
                     "environment": {k: v for k, v in sorted(env.items())
                                     if k.startswith(("GGML_", "CUDA_", "PI_MODEL", "LLAMA_PI0"))}}
            report["stages"][label] = stage
            write_json(out / "summary.json", report)
            print(f"\n阶段 {stage_number}: {label}", flush=True)
            with log_path.open("w") as log:
                child = subprocess.Popen(args_for_server, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                stage["pid"] = child.pid
                try:
                    h.wait_ready(child, opener)
                    need(child.poll() is None and h.listener_pid() == child.pid, "就绪服务与本次子进程不一致")
                    h.verify_server_identity(child.pid, binary, model, mmproj)
                    work(child, stage)
                    need(child.poll() is None and h.listener_pid() == child.pid, "阶段结束前服务进程身份发生变化")
                    stage["status"] = "requests_passed"
                except BaseException as exc:
                    stage.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                    raise
                finally:
                    try:
                        h.stop_child(child)
                    except BaseException as exc:
                        error = f"{label}: {type(exc).__name__}: {exc}"
                        report["cleanup_errors"].append(error)
                        stage["cleanup_error"] = error
                        raise
                    finally:
                        write_json(out / (label + ".json"), evidence_json(stage))
                        write_json(out / "summary.json", evidence_json(report))
            text = log_path.read_text(errors="replace")
            if trace:
                stage.update(verify_model_trace(text, mode, graphs))
            else:
                need("_shadow:" not in text and "flashrt_gemm:" not in text,
                     "性能阶段仍有 GEMM TRACE 或 shadow 输出")
            stage["status"] = "passed"
            write_json(out / (label + ".json"), stage)
            write_json(out / "summary.json", report)
            return stage

        for graphs in (False, True):
            sequence = [0] * 6 + [1] * 6 + [2] * 6 + [0] * 4
            for mode in (0, 2):
                label = f"e2e-graphs-{'on' if graphs else 'off'}-mode-{mode}"
                def validate(child, stage):
                    for i, case in enumerate(sequence, 1):
                        expected = targets.get(case)
                        need(mode == 0 or expected is not None, "候选请求缺少本次对应输入的对照")
                        record = infer(child, stage, case, expected, "correctness", i)
                        if mode == 0 and expected is None:
                            targets[case] = record["response"]
                        stage["records"].append(record)
                        print(f"{label}: {i}/{len(sequence)}，{cases[case]['id']} 动作有限且一致", flush=True)
                stage = run_server(label, mode, graphs, validate)
                print(f"{label} 通过；GEMM TRACE={stage['gemm_trace_count']}，"
                      f"graph warmup observed={stage['graph_warmup_complete_observed']}", flush=True)
        report["correctness"] = "exact_for_three_input_cases_in_both_graph_settings"
        report["mode2_status"] = "model_correctness_passed_for_tested_inputs"
        write_json(out / "summary.json", report)

        if not args.skip_performance:
            for index, mode in enumerate((0, 2, 2, 0), 1):
                label = f"perf-{index}-mode-{mode}"
                def measure(child, stage):
                    for i in range(1, 6):
                        record = infer(child, stage, 0, targets[0], "warmup", i)
                        stage["records"].append(record)
                        print(f"ABBA {index}/4 warmup {i}/5", flush=True)
                    for i in range(1, 21):
                        record = infer(child, stage, 0, targets[0], "measure", i)
                        stage["records"].append(record)
                        print(f"ABBA {index}/4 measure {i}/20: {record['infer_http_ms']:.3f} ms", flush=True)
                stage = run_server(label, mode, True, measure, trace=False)
                samples = []
                for record in stage["records"]:
                    if record["phase"] != "measure":
                        continue
                    response = record["response"]
                    values = {k: response[k] for k in METRICS[:4]}
                    values["handler_total_ms"] = response["timing_breakdown_ms"]["handler_total_ms"]
                    values["infer_http_ms"] = record["infer_http_ms"]
                    need(all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in values.values()),
                         "性能响应中的时延字段无效")
                    samples.append(values)
                need(len(samples) == 20, "性能阶段未获得 20 个有效测量样本")
                report["runs"].append({"mode": mode, "label": label, "samples": samples})
                write_json(out / "summary.json", report)
            performance = {"A": "accepted GGML-math LayerNorm + RMSNorm; GEMM=0",
                           "B": "same norm settings + GGML-mma16 GEMM=2",
                           "order": [0, 2, 2, 0], "metrics": {},
                           "scope": "fixed-input warm inference, reset/image/state requests excluded from HTTP timer"}
            report["performance"] = performance
            print("\nmetric,A_p50_ms,B_p50_ms,A_p95_ms,B_p95_ms,p50_latency_reduction_pct", flush=True)
            for key in METRICS:
                a = [s[key] for r in report["runs"] if r["mode"] == 0 for s in r["samples"]]
                b = [s[key] for r in report["runs"] if r["mode"] == 2 for s in r["samples"]]
                ap, bp = percentile(a, .5), percentile(b, .5)
                aq, bq = percentile(a, .95), percentile(b, .95)
                reduction = 100 * (ap - bp) / ap
                performance["metrics"][key] = {"A_p50": ap, "B_p50": bp, "A_p95": aq, "B_p95": bq,
                                               "p50_latency_reduction_pct": reduction}
                print(f"{key},{ap:.3f},{bp:.3f},{aq:.3f},{bq:.3f},{reduction:.2f}", flush=True)
            print("A/B 均保留已验收 norm；这里只比较增加 GEMM=2 的收益。正百分比表示耗时降低。", flush=True)
            report["mode2_status"] = "model_correctness_passed_and_benchmarked_for_tested_inputs"
        else:
            report["performance"] = {"status": "skipped_by_request"}
        need(manifest["sources"] == micro_mod.source_hashes(root) and
             manifest["artifacts"] == micro_mod.artifact_hashes(build), "E2E 期间源码或构建产物发生变化")
        need(manifest["cache_sha256"] == sha256(build / "CMakeCache.txt"), "E2E 期间 CMake 配置发生变化")
        need(report["models"] == [file_identity(model), file_identity(mmproj)] and
             report["input"]["image_sha256"] == sha256(image), "E2E 期间模型或图片发生变化")
        need(not report["cleanup_errors"], "本次服务清理未完成")
        report["status"] = "gemm_model_stage_complete"
        return 0
    except BaseException as exc:
        report.update(status="incomplete", error=f"{type(exc).__name__}: {exc}")
        print("阶段未完成:", report["error"], flush=True)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1
    finally:
        write_json(out / "summary.json", evidence_json(report))
        print("status:", report["status"], flush=True)
        print("correctness:", report.get("correctness", "未完成"), flush=True)
        print("已保存:", out / "summary.json", flush=True)
        print("本阶段覆盖 5 种短序列 GEMM；完整 kernel 移植仍未完成。", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
