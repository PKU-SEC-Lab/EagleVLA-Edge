#!/usr/bin/env python3
"""Validate the SM89 FlashRT bias/residual increment and measure its ABBA delta.

The accepted LayerNorm, RMSNorm and GEMM=2 paths remain enabled in both arms.
An older GEMM E2E result supplies action references, never build authorization.
Only subprocesses created by this invocation may be stopped.
"""

import argparse
import array
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import types
import urllib.request


ACTION_KEYS = ("action", "action_final_raw", "action_final")
METRICS = ("vit_ms", "encode_ms", "decode_ms", "total_ms",
           "handler_total_ms", "infer_http_ms")
TARGETS = ("llama-server", "test-flashrt-bias-residual")
REFERENCE = "results/flashrt-sm89/gemm-mma16-e2e-9__3vz_m/summary.json"


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def safe_json(value):
    return json.loads(json.dumps(value), parse_constant=lambda token: token)


def write_json(path, value):
    path = Path(path)
    pending = path.with_name(path.name + ".tmp")
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    pending.replace(path)


def percentile(values, q):
    values = sorted(values)
    need(bool(values), "没有有效时延样本")
    position = (len(values) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def wait_for_idle_gpu(gpu, output, micro, *, max_queries=15, interval=2.0):
    """Require two consecutive unchanged idle gates, retaining every observation.

    Only a validated no-process, low-memory snapshot whose utilization is still
    above 5% may be retried after gpu_snapshot raises RuntimeError. Query errors,
    active compute processes, high memory use and device changes fail at once.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "checking", "requested_gpu": str(gpu), "pinned_uuid": None,
              "max_queries": max_queries, "interval_seconds": interval,
              "required_consecutive_passes": 2, "attempts": [],
              "thresholds": {"compute_capability": "8.9", "compute_processes": 0,
                             "maximum_memory_used_mib": 256, "maximum_utilization_pct": 5}}
    summary = output / "idle-check-summary.json"
    consecutive = 0

    def inspect_snapshot(snapshot):
        need(isinstance(snapshot, dict), "GPU 空闲检查快照格式错误")
        uuid = snapshot.get("uuid")
        need(isinstance(uuid, str) and uuid.startswith("GPU-") and len(uuid) > 4,
             "GPU 空闲检查快照缺少有效 UUID")
        if report["pinned_uuid"] is None:
            report["pinned_uuid"] = uuid
        need(uuid == report["pinned_uuid"], "GPU 空闲检查期间设备 UUID 发生变化；未启动测试")
        need(snapshot.get("compute_capability") == "8.9", "GPU 空闲检查只支持 SM89")
        apps = snapshot.get("compute_processes")
        memory, utilization = snapshot.get("memory_used_mib"), snapshot.get("utilization_pct")
        need(isinstance(apps, list) and all(type(v) in (int, float) and math.isfinite(v)
                                          for v in (memory, utilization)),
             "GPU 空闲检查快照的进程/显存/利用率字段无效")
        need(memory >= 0 and 0 <= utilization <= 100, "GPU 空闲检查快照的显存/利用率范围无效")
        need(not apps and memory <= 256,
             f"GPU {uuid} 有计算进程或显存占用（进程 {len(apps)} 个，显存 {memory:g} MiB）；"
             "未停止任何进程，也未启动新测试")
        return utilization <= 5

    try:
        need(type(max_queries) is int and 2 <= max_queries <= 15,
             "GPU 空闲检查最多 15 次，且需要至少两次连续通过")
        need(type(interval) in (int, float) and math.isfinite(interval) and interval >= 2,
             "GPU 空闲检查的采样间隔不得小于 2 秒")
        for number in range(1, max_queries + 1):
            attempt_dir = output / f"idle-check-{number:02d}"
            attempt_dir.mkdir(exist_ok=False)
            attempt = {"number": number, "directory": str(attempt_dir),
                       "started_wall_ns": time.time_ns(), "status": "querying"}
            report["attempts"].append(attempt)
            write_json(summary, report)
            try:
                try:
                    snapshot = micro.gpu_snapshot(report["pinned_uuid"] or gpu, attempt_dir)
                except RuntimeError as error:
                    attempt["original_error"] = f"{type(error).__name__}: {error}"
                    snapshot_path = attempt_dir / "gpu.json"
                    if not snapshot_path.is_file():
                        raise
                    snapshot = json.loads(snapshot_path.read_text())
                    attempt["snapshot"] = snapshot
                    idle = inspect_snapshot(snapshot)
                    if idle:
                        # An apparently idle snapshot cannot explain a failed query;
                        # retain the original failure instead of silently retrying.
                        raise
                    consecutive = 0
                    attempt["status"] = "waiting_for_utilization"
                else:
                    attempt["snapshot"] = snapshot
                    need(inspect_snapshot(snapshot), "GPU helper 返回成功，但利用率门槛未通过")
                    consecutive += 1
                    attempt["status"] = "idle_gate_passed"
                attempt["consecutive_passes"] = consecutive
                attempt["finished_wall_ns"] = time.time_ns()
                write_json(summary, report)
                if consecutive == 2:
                    report["status"] = "idle_confirmed"
                    report["final_snapshot"] = snapshot
                    write_json(output / "gpu.json", snapshot)
                    print(f"GPU {report['pinned_uuid']} 连续两次通过空闲检查。", flush=True)
                    return snapshot
                if number < max_queries:
                    print(f"GPU 空闲检查 {number}/{max_queries}：连续通过 {consecutive}/2；"
                          f"{interval:g} 秒后复查。", flush=True)
                    time.sleep(interval)
            except BaseException as error:
                attempt.update(status="failed", error=f"{type(error).__name__}: {error}",
                               finished_wall_ns=time.time_ns())
                raise
        raise RuntimeError(f"GPU {report['pinned_uuid'] or gpu} 在 {max_queries} 次检查内未连续两次空闲；"
                           "未停止任何进程，也未启动下一阶段。请稍后重试。")
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["finished_wall_ns"] = time.time_ns()
        write_json(summary, safe_json(report))


def source_hashes(root, micro):
    sources = micro.source_hashes(root)
    for path in sorted((root / "vendor").rglob("*")):
        if path.is_file() and (path.name == "CMakeLists.txt" or path.suffix in
                              {".c", ".cc", ".cpp", ".cu", ".h", ".hpp", ".cuh", ".cmake"}):
            sources[str(path.relative_to(root))] = sha256(path)
    return sources


def artifact_hashes(build):
    paths = {build / "bin" / target for target in TARGETS}
    paths.update(p for p in (build / "bin").glob("lib*.so*") if p.is_file())
    need(all(path.is_file() for path in paths), "缺少本阶段构建产物")
    need(any(path.name.startswith("libggml-cuda") for path in paths), "缺少 CUDA 后端动态库")
    return {str(p.relative_to(build)): sha256(p) for p in sorted(paths)}


def build_stage(args, micro, out):
    cmake = shutil.which("cmake")
    need(cmake, "找不到 cmake")
    before = source_hashes(args.root, micro)
    micro.execute([cmake, "-S", args.root, "-B", args.build,
                   "-DGGML_CUDA=ON", "-DGGML_CUDA_FLASHRT=ON",
                   "-DCMAKE_CUDA_ARCHITECTURES=89", "-DLLAMA_BUILD_TESTS=ON"], out / "configure.log")
    micro.execute([cmake, "--build", args.build, "--target", *TARGETS, "-j", args.jobs],
                  out / "build.log", timeout=args.build_timeout)
    after = source_hashes(args.root, micro)
    need(before == after, "构建期间源码发生变化，不能验收混合版本")
    manifest = {"root": str(args.root), "build": str(args.build), "sources": after,
                "artifacts": artifact_hashes(args.build),
                "cache_sha256": sha256(args.build / "CMakeCache.txt")}
    write_json(out / "build-manifest.json", manifest)
    return manifest


def environment(gpu, mode, graphs, trace, helper_env=None, graph_trace=False):
    env = dict(os.environ)
    env.update(helper_env or {})
    for key in list(env):
        if key.startswith("GGML_CUDA_FLASHRT"):
            env.pop(key)
    for key in ("PI0_ACTION_NOISE_BIN", "PI0_NORM_STATS_JSON", "PI_MODEL_DEBUG_DUMP_FILE",
                "PI05_DEBUG_DUMP_FILE", "GGML_CUDA_DISABLE_GRAPHS"):
        env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES=gpu, PI_MODEL="pi0", GGML_CUDA_FLASHRT="1",
               GGML_CUDA_FLASHRT_GGML_MATH="1", GGML_CUDA_FLASHRT_RMSNORM="1",
               GGML_CUDA_FLASHRT_GEMM="2", GGML_CUDA_FLASHRT_BIAS_RESIDUAL=str(mode),
               GGML_CUDA_FLASHRT_SHADOW="0", GGML_CUDA_FLASHRT_GEMM_SHADOW="0",
               GGML_CUDA_FLASHRT_GEMM_GRAPH_TRACE="0", GGML_CUDA_FLASHRT_TRACE=str(int(trace)),
               GGML_CUDA_FLASHRT_BIAS_GRAPH_TRACE=str(int(graph_trace)),
               GGML_CUDA_DISABLE_FUSION="0", PI_MODEL_DEBUG="0", LLAMA_PI0_PERF="0")
    if not graphs:
        env["GGML_CUDA_DISABLE_GRAPHS"] = "1"
    return env


def flat_action(response, key):
    rows = response.get(key)
    need(isinstance(rows, list) and len(rows) == 10, f"{key}: 动作应为 10x32")
    need(all(isinstance(row, list) and len(row) == 32 for row in rows), f"{key}: 动作应为 10x32")
    values = [value for row in rows for value in row]
    need(all(type(value) in (int, float) and math.isfinite(value) for value in values),
         f"{key}: 包含无效值")
    return values


def compare_actions(actual, expected):
    for key in ACTION_KEYS:
        a, b = flat_action(actual, key), flat_action(expected, key)
        diffs = [u - v for u, v in zip(a, b)]
        maximum = max(abs(d) for d in diffs)
        rmse = math.sqrt(math.fsum(d * d for d in diffs) / len(diffs))
        need(maximum == 0 and rmse == 0, f"{key}: max_abs={maximum:.9g}, RMSE={rmse:.9g}")


def reference_actions(reference, cases):
    need(reference.get("status") == "gemm_model_stage_complete" and
         reference.get("correctness") == "exact_for_three_input_cases_in_both_graph_settings",
         "旧 GEMM E2E 记录未完成规定的数值验收")
    need(reference.get("input", {}).get("cases") == cases,
         "旧 GEMM E2E 的三个输入案例与本阶段不一致")
    wanted = {case["id"] for case in cases}
    targets = {}
    expected_stages = {f"e2e-graphs-{graphs}-mode-{mode}" for graphs in ("off", "on") for mode in (0, 2)}
    stages = {label: stage for label, stage in reference.get("stages", {}).items() if label.startswith("e2e-")}
    need(set(stages) == expected_stages, "旧 GEMM E2E 缺少四种模型设置")
    sequence = ["fixed"] * 6 + ["changed-text"] * 6 + ["changed-state"] * 6 + ["fixed"] * 4
    for label, stage in stages.items():
        need(stage.get("status") == "passed", "旧 GEMM E2E 包含未通过的模型阶段")
        need(stage.get("label") == label and [r.get("case") for r in stage.get("records", [])] == sequence,
             "旧 GEMM E2E 的请求案例/顺序不完整")
        for record in stage.get("records", []):
            case = record.get("case")
            need(case in wanted, "旧 GEMM E2E 包含未知案例")
            response = record["response"]
            for key in ACTION_KEYS:
                flat_action(response, key)
            if case in targets:
                compare_actions(response, targets[case])
            else:
                targets[case] = response
    need(set(targets) == wanted, "旧 GEMM E2E 缺少三个案例的动作参考")
    return targets


class Telemetry:
    """Owned nvidia-smi children plus a CPU sampler; no system settings changed."""

    def __init__(self, gpu, directory):
        self.gpu, self.directory = gpu, directory
        self.children, self.streams, self.errors = [], [], []
        self.stop_event = threading.Event()
        self.thread = None
        self.started = None

    def start(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.started = time.time_ns()
        nvidia = shutil.which("nvidia-smi")
        need(nvidia, "性能遥测找不到 nvidia-smi")
        queries = {
            "gpu": "--query-gpu=timestamp,index,uuid,memory.used,utilization.gpu,utilization.memory,power.draw,clocks.current.sm,clocks.current.memory",
            "processes": "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        }
        try:
            for label, query in queries.items():
                stream = (self.directory / f"{label}.csv").open("w")
                self.streams.append(stream)
                child = subprocess.Popen([nvidia, "-i", self.gpu, query, "--format=csv,noheader,nounits", "--loop=1"],
                                         stdin=subprocess.DEVNULL, stdout=stream,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                self.children.append(child)
            self.thread = threading.Thread(target=self._sample_cpu, daemon=True)
            self.thread.start()
            self.check()
        except BaseException:
            self.stop()
            raise

    def _sample_cpu(self):
        try:
            with (self.directory / "cpu.jsonl").open("w") as output:
                while not self.stop_event.is_set():
                    data = {"wall_time_ns": time.time_ns(), "monotonic_ns": time.monotonic_ns(),
                            "loadavg": list(os.getloadavg()), "cpu_count": os.cpu_count()}
                    for source, key in (("/proc/stat", "proc_stat"), ("/proc/pressure/cpu", "cpu_pressure")):
                        try:
                            text = Path(source).read_text()
                            data[key] = text.splitlines()[0] if key == "proc_stat" else text.strip()
                        except OSError as error:
                            data[key + "_error"] = str(error)
                    output.write(json.dumps(data, allow_nan=False) + "\n")
                    output.flush()
                    self.stop_event.wait(1)
        except BaseException as error:
            self.errors.append(f"CPU sampler: {type(error).__name__}: {error}")

    def check(self):
        for child in self.children:
            need(child.poll() is None, f"nvidia-smi 遥测提前退出 ({child.returncode})；查看 {self.directory}")
        need(not self.errors, "; ".join(self.errors))

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=3)
            if self.thread.is_alive():
                self.errors.append("CPU sampler 未按期退出")
        for child in self.children:
            try:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=5)
            except BaseException as error:
                self.errors.append(f"telemetry PID {child.pid}: {type(error).__name__}: {error}")
        for stream in self.streams:
            stream.close()
        result = {"directory": str(self.directory), "started_wall_ns": self.started,
                  "ended_wall_ns": time.time_ns(), "owned_pids": [c.pid for c in self.children],
                  "cleanup_errors": list(self.errors),
                  "scope": "1-second GPU/process samples and host load; no causal attribution",
                  "process_sample_time_scope": "processes.csv has no per-sample timestamps or iteration boundaries; "
                                               "it identifies processes observed within a block and cannot align "
                                               "a process observation with one slow request"}
        write_json(self.directory / "summary.json", result)
        return result


def trace_records(text):
    return [dict(re.findall(r"(\w+)=([^\s]+)", line)) for line in text.splitlines()
            if "flashrt_bias_residual:" in line]


def verify_model_trace(text, mode, graphs):
    need("_shadow:" not in text, "模型验收出现 shadow，未验证候选真实输出传播")
    records = trace_records(text)
    if mode == 0:
        need(not records, "BIAS_RESIDUAL=0 对照命中了候选")
    else:
        need(bool(records), "模型没有命中 bias/residual 候选；不能把 fallback 当作移植完成")
        need(all(r.get("pattern") in ("direct", "cont") and
                 int(r.get("width", 0)) > 0 and int(r.get("rows", 0)) > 0 and
                 bool(r.get("node")) and r.get("capture") in ("0", "1") for r in records),
             "bias/residual TRACE 字段缺失或无效")
        need({r["pattern"] for r in records} == {"direct", "cont"},
             "真实模型未同时命中 direct 和 cont；不能把整条路径 fallback 当作完成")
        need(graphs or all(r["capture"] == "0" for r in records), "关闭 Graph 阶段出现 capture")
        if graphs:
            need({r["pattern"] for r in records if r["capture"] == "1"} == {"direct", "cont"},
                 "真实模型 Graph capture 未同时覆盖 direct 和 cont")
    warmups = text.count("CUDA graph warmup complete")
    need((warmups > 0) == graphs, "模型 Graph 设置与 warmup 日志不符")
    return {"bias_trace_count": len(records), "patterns": sorted({r["pattern"] for r in records}),
            "pattern_counts": {p: sum(r["pattern"] == p for r in records) for p in ("direct", "cont")},
            "distinct_nodes": sorted({r.get("node", "") for r in records}),
            "shapes": sorted({(int(r["width"]), int(r["rows"])) for r in records}),
            "captured_dispatches": sum(r.get("capture") == "1" for r in records),
            "graph_warmup_observed": warmups,
            "model_graph_replay_count": None,
            "graph_evidence_scope": "model dispatch capture/warmup; fixture actual replay checked independently"}


def performance_report(runs):
    result = {"A": "accepted norm and GEMM=2, bias/residual=0",
              "B": "same accepted paths plus bias/residual=1", "order": [0, 1, 1, 0],
              "metrics": {}, "blocks": [], "all_samples_retained": True,
              "scope": "fixed-image warm inference; HTTP timer excludes reset/image/state requests"}
    need([r["mode"] for r in runs] == result["order"], "ABBA 进程顺序不完整")
    for run in runs:
        need(len(run["samples"]) == 20, "性能区块应保留全部 20 次样本")
        values = {}
        for key in METRICS:
            samples = [s[key] for s in run["samples"]]
            need(all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in samples),
                 f"{run['label']} {key} 包含无效耗时")
            values[key] = {"p50": percentile(samples, .5), "p95": percentile(samples, .95),
                           "minimum": min(samples), "maximum": max(samples)}
        result["blocks"].append({"label": run["label"], "mode": run["mode"], "metrics": values})
    for key in METRICS:
        a = [s[key] for r in runs if r["mode"] == 0 for s in r["samples"]]
        b = [s[key] for r in runs if r["mode"] == 1 for s in r["samples"]]
        ap, bp = percentile(a, .5), percentile(b, .5)
        result["metrics"][key] = {"A_p50": ap, "B_p50": bp, "A_p95": percentile(a, .95),
                                  "B_p95": percentile(b, .95), "p50_latency_reduction_pct": 100 * (ap - bp) / ap}
    return result


def fixture_result(prefix, mode, graphs, text):
    data = json.loads(prefix.with_suffix(".json").read_text())
    need(data.get("schema_version") == 1 and data.get("status") == "tests_complete" and
         data.get("backend") == "CUDA0" and data.get("mode") == mode and
         data.get("graphs_requested") is graphs, "fixture JSON 设置或完成状态不符")
    need(type(data.get("host_checks")) is int and data["host_checks"] >= 69,
         "fixture 未执行主机契约检查")
    need(data.get("raw_dtype") == "little-endian-f32" and
         Path(data.get("raw_file", "")).resolve() == prefix.with_suffix(".bin").resolve(),
         "fixture 原始输出格式/路径不符合协议")
    need(data.get("gpu_alias_cases") == 0,
         "alias 拒绝契约必须在主机检查；GPU fallback fixture 不得包含未定义 alias")
    records = data.get("records")
    need(isinstance(records, list) and len(records) == data.get("case_count") and len(records) > 0,
         "fixture JSON 的案例集合缺失")
    names = [record.get("case") for record in records]
    need(len(set(names)) == len(names) and all(isinstance(n, str) and n for n in names),
         "fixture 案例名称无效或重复")
    expected = {}
    for pattern in ("direct", "cont"):
        for rows in (1, 50, 256, 512):
            expected[f"{pattern}_r{rows}"] = (pattern, rows, True, "normal")
        for suffix, rows in (("cancellation", 50), ("signed-zero", 50), ("bias-first", 1),
                             ("residual-first", 50), ("both-first", 1)):
            expected[f"{pattern}_{suffix}"] = (pattern, rows, True, "normal")
        for variant in ("extra-consumer", "intermediate-output", "wrong-edge", "strided-x", "wrong-bias-shape"):
            expected[f"{pattern}_fallback_{variant}"] = (pattern, 50, False, variant)
    for variant in ("wrong-cont-edge", "cont-output"):
        expected[f"cont_fallback_{variant}"] = ("cont", 50, False, variant)
    need(set(names) == set(expected) and len(names) == 30, "fixture 未覆盖规定的 18 个命中图和 12 个拒绝图")
    raw = prefix.with_suffix(".bin").read_bytes()
    traces = trace_records(text)
    launches = re.findall(r"flashrt_bias_graph:\s+launch\s+node=([^\s]+)", text)
    allowed_nodes = {record["node"] for record in records}
    allowed_graphs = {record["graph_first_node"] for record in records}
    need(all(r.get("node") in allowed_nodes for r in traces), "fixture 候选 TRACE 包含未知节点")
    need(all(name in allowed_graphs for name in launches), "fixture replay TRACE 包含未知图")
    if not graphs:
        need(not launches, "Graphs off fixture 出现实际 cudaGraphLaunch")
    cursor, decoded = 0, {}
    identities = []
    for record in records:
        name = record["case"]
        count = record.get("elements")
        need((record.get("pattern"), record.get("rows"), record.get("eligible"), record.get("variant")) == expected[name]
             and record.get("width") == 1152 and count == 1152 * expected[name][1]
             and record.get("node") == f"bias_test_{name}_out"
             and record.get("graph_first_node") == f"bias_test_{name}_first",
             f"{name}: fixture 元数据与规定覆盖不符")
        need(type(count) is int and count > 0 and record.get("offset_elements") == cursor and
             record.get("raw_offset_bytes") == cursor * 4, f"{name}: binary 布局异常")
        need(type(record.get("eligible")) is bool and
             record.get("expected_candidate") is (bool(mode) and record["eligible"]),
             f"{name}: 候选资格/期望命中字段错误")
        need(record.get("finite") is True and record.get("reference_bit_mismatches") == 0 and
             record.get("repeat_bit_mismatches") == 0 and record.get("repeat_max_abs") == 0,
             f"{name}: 参考或重复输出位模式检查失败")
        need(record.get("compute_calls") == (12 if graphs else 2), f"{name}: 计算次数不符合协议")
        piece = raw[cursor * 4:(cursor + count) * 4]
        need(len(piece) == count * 4, f"{name}: F32 输出截断")
        values = array.array("f")
        values.frombytes(piece)
        if sys.byteorder != "little":
            values.byteswap()
        need(values.itemsize == 4 and all(math.isfinite(v) for v in values), f"{name}: 非有限 F32 输出")
        hits = [r for r in traces if r.get("node") == record["node"]]
        if record["expected_candidate"]:
            need(bool(hits), f"{name}: 合格图未真正命中候选")
            need(all(r.get("pattern") == record["pattern"] and
                     int(r.get("width", 0)) == record["width"] and
                     int(r.get("rows", 0)) == record["rows"] and
                     r.get("capture") in ("0", "1") for r in hits),
                 f"{name}: 候选 TRACE 与 fixture 元数据不符")
            if graphs:
                need(any(r["capture"] == "1" for r in hits), f"{name}: 未观察候选 capture")
            else:
                need(all(r["capture"] == "0" for r in hits), f"{name}: 关闭 Graph 后仍 capture")
        else:
            need(not hits, f"{name}: 不合格/关闭候选的图发生了候选命中")
        replay_count = launches.count(record["graph_first_node"])
        need(not graphs or replay_count >= 6, f"{name}: 仅观察 {replay_count} 次实际 Graph replay，至少需要 6")
        record["actual_candidate_dispatches"] = len(hits)
        record["actual_graph_launches"] = replay_count
        decoded[name] = piece
        identities.append({k: record[k] for k in ("case", "pattern", "node", "graph_first_node", "variant",
                                                  "width", "rows", "eligible", "offset_elements", "elements")})
        cursor += count
    need(len(raw) == cursor * 4, "fixture binary 存在未登记尾部数据")
    data["identities"] = identities
    data["binary_sha256"] = sha256(prefix.with_suffix(".bin"))
    return data, decoded


def run_fixtures(args, gpu, micro, report, out, helper):
    test = args.build / "bin/test-flashrt-bias-residual"
    _, host_text = micro.execute([test, "--host-only"], out / "host-contracts.log")
    count = re.search(r"(\d+) bias/residual contract checks passed", host_text)
    need(count is not None and int(count[1]) >= 69, "未观察到全部 69 项主机契约检查通过")
    report["host_contract_checks"] = int(count[1])
    reference = None
    reference_identity = None
    fixture_runs = []
    report["fixtures"] = fixture_runs
    for graphs in (False, True):
        for mode in (0, 1):
            label = f"fixtures-graphs-{'on' if graphs else 'off'}-mode-{mode}"
            need(helper.listener_pid() is None, "18081 已被占用；没有启动新的测试进程")
            snapshot_dir = out / (label + "-device")
            snapshot_dir.mkdir()
            snapshot = wait_for_idle_gpu(gpu, snapshot_dir, micro)
            need(snapshot["uuid"] == gpu, "fixture GPU 身份发生变化")
            prefix = out / label
            _, text = micro.execute([test, "--mode", mode, "--graphs", "on" if graphs else "off",
                                     "--output-prefix", prefix], prefix.with_suffix(".log"),
                                    env=environment(gpu, mode, graphs, True, graph_trace=True))
            data, values = fixture_result(prefix, mode, graphs, text)
            if reference is None:
                reference, reference_identity = values, data["identities"]
            need(data["identities"] == reference_identity and set(values) == set(reference),
                 "独立进程的 fixture 案例/形状/顺序不一致")
            need(all(values[name] == reference[name] for name in reference),
                 f"{label}: 输出二进制与本阶段 GGML 对照不一致（包含符号零检查）")
            data["binary_equal_to_graphs_off_mode0"] = True
            write_json(prefix.with_name(prefix.name + "-checked.json"), data)
            fixture_runs.append({"label": label, "mode": mode, "graphs": graphs,
                                 "checked_json": str(prefix.with_name(prefix.name + "-checked.json")),
                                 "case_count": len(values), "binary_equal": True,
                                 "minimum_actual_graph_launches_per_case": min(
                                     r["actual_graph_launches"] for r in data["records"])})
            write_json(out / "summary.json", report)
            print(f"{label}: {len(values)} cases，binary exact；Graph 调用协议检查通过", flush=True)
    report["fixture_correctness"] = "binary_exact_in_four_independent_processes"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--build", type=Path)
    parser.add_argument("--reference-e2e", type=Path)
    parser.add_argument("--gpu", help="物理 GPU index 或 UUID；默认沿用已验收 GEMM 的 GPU UUID")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--build-timeout", type=int, default=1800)
    parser.add_argument("--skip-performance", action="store_true")
    args = parser.parse_args(argv)
    args.root = args.root.resolve(strict=True)
    args.build = (args.build or args.root / "build-flashrt-sm89").resolve()
    reference_path = (args.reference_e2e or args.root / REFERENCE).resolve()
    results = args.root / "results/flashrt-sm89"
    results.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="bias-residual-stage-", dir=results))
    report = {"schema_version": 1, "status": "running", "full_port_complete": False,
              "scope": "FlashRT F32 two-ADD bias/residual fusion, direct and CONT patterns",
              "runner_sha256": sha256(Path(__file__)), "stages": {}, "runs": [], "cleanup_errors": [],
              "reference_e2e": str(reference_path)}
    print("本次结果目录:", out, flush=True)
    try:
        need(args.jobs > 0 and args.build_timeout > 0, "jobs/build-timeout 必须为正整数")
        need(Path("/proc/self/stat").read_text().split(" ", 1)[0] == str(os.getpid()),
             "procfs 与当前 PID 不在同一命名空间；请在 gpu4 普通 SSH 终端执行")
        micro_path = args.root / "scripts/run-sm89-gemm-stage.py"
        helper_path = args.root / "run-ggml-math-shadow.py"
        micro = types.SimpleNamespace(**runpy.run_path(str(micro_path)))
        h = types.SimpleNamespace(**runpy.run_path(str(helper_path)))
        need(not hasattr(h, "ROOT") or Path(h.ROOT).resolve() == args.root, "服务 helper 指向其他仓库")
        report["helper_sha256"] = sha256(helper_path)
        report["micro_helper_sha256"] = sha256(micro_path)
        need(h.listener_pid() is None, "18081 已被占用；未停止任何已有进程，也未启动测试")
        reference = json.loads(reference_path.read_text())
        report["reference_sha256"] = sha256(reference_path)
        cases = [{"id": "fixed", "text": h.PROMPT, "state": "0," * 31 + "0"},
                 {"id": "changed-text", "text": "move the object to the left", "state": "0," * 31 + "0"},
                 {"id": "changed-state", "text": h.PROMPT, "state": ",".join(map(str, [.1, -.1] + [0] * 30))}]
        targets = reference_actions(reference, cases)
        image = Path(h.IMAGE).resolve(strict=True)
        need(reference["input"].get("image_sha256") == sha256(image), "本阶段图片与旧 GEMM 参考不一致")
        report["input"] = {"image": str(image), "image_sha256": sha256(image), "cases": cases}
        report["reference_scope"] = "action regression only; current build is independently frozen and validated"
        gpu_request = args.gpu or reference.get("gpu", {}).get("uuid")
        need(gpu_request, "旧 E2E 缺少 GPU UUID；请用 --gpu 指定空闲的 4090")
        report["gpu"] = wait_for_idle_gpu(gpu_request, out, micro)
        gpu = report["gpu"]["uuid"]
        manifest = build_stage(args, micro, out)
        report["build"] = {"directory": str(args.build), "manifest": str(out / "build-manifest.json"),
                           "manifest_sha256": sha256(out / "build-manifest.json")}
        write_json(out / "summary.json", report)
        run_fixtures(args, gpu, micro, report, out, h)
        binary = (args.build / "bin/llama-server").resolve(strict=True)
        model = (Path(h.MODEL_DIR) / "pi_llm.gguf").resolve(strict=True)
        mmproj = (Path(h.MODEL_DIR) / "mmproj-model-f16.gguf").resolve(strict=True)

        def identity(path):
            stat = path.stat()
            return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                    "device": stat.st_dev, "inode": stat.st_ino}

        model_identity = [identity(model), identity(mmproj)]
        need(reference.get("models") == model_identity, "当前模型文件身份与已验收 GEMM 参考不一致")
        report["models"] = model_identity
        report["model_sha256"] = {str(path): sha256(path) for path in (model, mmproj)}
        command = [str(binary), "-m", str(model), "--mmproj", str(mmproj), "-ngl", "99",
                   "-c", "2048", "--parallel", "1", "--flash-attn", "on", "--no-warmup",
                   "--host", "127.0.0.1", "--port", "18081", "-lv", "5"]
        report["command"] = command
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

        def infer(child, stage, case_index, phase, index):
            case = cases[case_index]
            need(child.poll() is None and h.listener_pid() == child.pid, "请求前服务身份发生变化")
            need(h.request(opener, "POST", "/foreground/reset").get("reset") is True, "reset 失败")
            need(h.request(opener, "POST", "/foreground/image", {"path": str(image)}, timeout=30).get("loaded") is True,
                 "图片加载失败")
            state = h.request(opener, "PUT", "/foreground/state", {"state": case["state"]})
            need(state.get("loaded") is True and state.get("state_dim") == 32, "state 加载失败")
            wall = time.time_ns()
            started = time.perf_counter_ns()
            response = h.request(opener, "POST", "/foreground/infer", {"text": case["text"]}, timeout=120)
            ended = time.perf_counter_ns()
            record = {"case": case["id"], "phase": phase, "request_index": index,
                      "started_wall_ns": wall, "started_monotonic_ns": started, "ended_monotonic_ns": ended,
                      "infer_http_ms": (ended - started) / 1e6, "response": response}
            stage["records"].append(record)
            try:
                need(child.poll() is None and h.listener_pid() == child.pid, "请求后服务身份发生变化")
                compare_actions(response, targets[case["id"]])
            except BaseException:
                failed = out / f"{stage['label']}-failed-response-{index}.json"
                write_json(failed, safe_json({"actual": record, "expected": targets[case["id"]]}))
                stage["failed_response"] = str(failed)
                raise
            return record

        def run_server(label, mode, graphs, work, trace=True):
            need(h.listener_pid() is None, "18081 已被占用；未启动新服务或停止已有进程")
            need([identity(model), identity(mmproj)] == model_identity and
                 sha256(image) == report["input"]["image_sha256"], "模型或图片在测试过程中发生变化")
            device_dir = out / (label + "-device")
            device_dir.mkdir()
            wait_for_idle_gpu(gpu, device_dir, micro)
            need(h.listener_pid() is None, "启动前 18081 已被占用；未启动新服务")
            env = environment(gpu, mode, graphs, trace, h.MODE_ENV)
            cmd = command[:]
            cmd[-1] = "5" if trace else "3"
            log_path = out / (label + ".log")
            stage = {"label": label, "mode": mode, "graphs_allowed": graphs, "trace": trace,
                     "status": "starting", "records": [], "log": str(log_path),
                     "environment": {k: v for k, v in env.items() if k.startswith(("GGML_", "CUDA_", "PI_MODEL", "LLAMA_PI0"))}}
            report["stages"][label] = stage
            write_json(out / "summary.json", report)
            print("阶段:", label, flush=True)
            with log_path.open("w") as log:
                child = subprocess.Popen(cmd, cwd=args.root, env=env, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                stage["pid"] = child.pid
                telemetry = None
                try:
                    h.wait_ready(child, opener)
                    need(child.poll() is None and h.listener_pid() == child.pid, "就绪服务不属于本次子进程")
                    h.verify_server_identity(child.pid, binary, model, mmproj)
                    if not trace:
                        telemetry = Telemetry(gpu, out / (label + "-telemetry"))
                        telemetry.start()
                    work(child, stage, telemetry)
                    if telemetry is not None:
                        telemetry.check()
                    need(child.poll() is None and h.listener_pid() == child.pid, "阶段结束前服务身份发生变化")
                    stage["status"] = "requests_passed"
                except BaseException as exc:
                    stage.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                    raise
                finally:
                    if telemetry is not None:
                        try:
                            stage["telemetry"] = telemetry.stop()
                            report["cleanup_errors"].extend(stage["telemetry"]["cleanup_errors"])
                        except BaseException as exc:
                            report["cleanup_errors"].append(f"{label} telemetry: {type(exc).__name__}: {exc}")
                    try:
                        h.stop_child(child)
                    except BaseException as exc:
                        report["cleanup_errors"].append(f"{label} server: {type(exc).__name__}: {exc}")
                    write_json(out / (label + ".json"), safe_json(stage))
                    write_json(out / "summary.json", safe_json(report))
            need(not report["cleanup_errors"], "本阶段拥有的进程清理失败：" + "; ".join(report["cleanup_errors"]))
            text = log_path.read_text(errors="replace")
            if trace:
                stage.update(verify_model_trace(text, mode, graphs))
            else:
                need("_shadow:" not in text and "flashrt_" not in text,
                     "性能阶段仍有 FlashRT trace/shadow 输出")
                need((out / (label + "-telemetry/gpu.csv")).stat().st_size > 0 and
                     (out / (label + "-telemetry/cpu.jsonl")).stat().st_size > 0,
                     "性能遥测没有采样记录")
                processes_path = out / (label + "-telemetry/processes.csv")
                need(processes_path.is_file() and processes_path.stat().st_size > 0,
                     f"性能进程遥测为空，不能确认区块内的计算进程；查看 {processes_path}")
                with processes_path.open() as stream:
                    process_rows = list(csv.reader(stream, skipinitialspace=True))
                need(any(len(row) == 4 and row[0].strip() == gpu and
                         row[1].strip() == str(child.pid) for row in process_rows),
                     f"性能进程遥测未记录本次服务 GPU={gpu} PID={child.pid}；查看 {processes_path}")
                stage["telemetry"]["owned_server_observed"] = True
            stage["status"] = "passed"
            write_json(out / (label + ".json"), stage)
            write_json(out / "summary.json", report)
            return stage

        sequence = [0] * 6 + [1] * 6 + [2] * 6 + [0] * 4
        for graphs in (False, True):
            for mode in (0, 1):
                label = f"e2e-graphs-{'on' if graphs else 'off'}-mode-{mode}"
                def validate(child, stage, telemetry):
                    for index, case_index in enumerate(sequence, 1):
                        infer(child, stage, case_index, "correctness", index)
                        print(f"{label}: {index}/22，{cases[case_index]['id']} 与已验收 GEMM 动作一致", flush=True)
                stage = run_server(label, mode, graphs, validate)
                print(f"{label} 通过；bias TRACE={stage['bias_trace_count']}，"
                      f"patterns={stage['pattern_counts']}，graph warmup={stage['graph_warmup_observed']}", flush=True)
        report["correctness"] = "exact_to_accepted_gemm_for_three_input_cases_in_both_graph_settings"
        report["correctness_requests"] = 88
        write_json(out / "summary.json", report)

        if not args.skip_performance:
            for block, mode in enumerate((0, 1, 1, 0), 1):
                label = f"perf-{block}-mode-{mode}"
                def measure(child, stage, telemetry):
                    for phase, count in (("warmup", 5), ("measure", 20)):
                        for index in range(1, count + 1):
                            telemetry.check()
                            record = infer(child, stage, 0, phase, index)
                            telemetry.check()
                            print(f"ABBA {block}/4 {phase} {index}/{count}: {record['infer_http_ms']:.3f} ms", flush=True)
                stage = run_server(label, mode, True, measure, trace=False)
                samples = []
                for record in stage["records"]:
                    if record["phase"] == "measure":
                        response = record["response"]
                        values = {key: response[key] for key in METRICS[:4]}
                        values.update(handler_total_ms=response["timing_breakdown_ms"]["handler_total_ms"],
                                      infer_http_ms=record["infer_http_ms"])
                        samples.append(values)
                report["runs"].append({"label": label, "mode": mode, "samples": samples,
                                       "telemetry": stage["telemetry"]})
                write_json(out / "summary.json", report)
            report["performance"] = performance_report(report["runs"])
            print("\nblock,mode,metric,p50_ms,p95_ms", flush=True)
            for block in report["performance"]["blocks"]:
                for key, values in block["metrics"].items():
                    print(f"{block['label']},{block['mode']},{key},{values['p50']:.3f},{values['p95']:.3f}", flush=True)
            print("\nmetric,A_p50_ms,B_p50_ms,A_p95_ms,B_p95_ms,p50_latency_reduction_pct", flush=True)
            for key, row in report["performance"]["metrics"].items():
                print(f"{key},{row['A_p50']:.3f},{row['B_p50']:.3f},{row['A_p95']:.3f},{row['B_p95']:.3f},"
                      f"{row['p50_latency_reduction_pct']:.2f}", flush=True)
            print("所有慢样本均保留；GPU/进程/CPU 遥测已保存，本报告不推断延迟波动原因。", flush=True)
        else:
            report["performance"] = {"status": "skipped_by_request"}
        need(manifest["sources"] == source_hashes(args.root, micro) and
             manifest["artifacts"] == artifact_hashes(args.build) and
             manifest["cache_sha256"] == sha256(args.build / "CMakeCache.txt"), "验收期间源码/构建产物/配置发生变化")
        need(model_identity == [identity(model), identity(mmproj)] and
             report["model_sha256"] == {str(path): sha256(path) for path in (model, mmproj)} and
             report["input"]["image_sha256"] == sha256(image), "验收期间模型或图片发生变化")
        need(not report["cleanup_errors"], "本阶段进程清理未完成")
        report["status"] = "bias_residual_model_stage_complete"
        return 0
    except BaseException as exc:
        report.update(status="incomplete", error=f"{type(exc).__name__}: {exc}")
        print("阶段未完成:", report["error"], flush=True)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1
    finally:
        write_json(out / "summary.json", safe_json(report))
        print("status:", report["status"], flush=True)
        print("correctness:", report.get("correctness", "未完成"), flush=True)
        print("已保存:", out / "summary.json", flush=True)
        print("本阶段验收 bias/residual direct 与 CONT 融合；完整 kernel 移植仍未完成。", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
