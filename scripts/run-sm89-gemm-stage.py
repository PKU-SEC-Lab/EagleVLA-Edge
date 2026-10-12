#!/usr/bin/env python3
"""Validate the experimental SM89 GEMM candidates in separate GPU processes.

This runner does not start, stop, or contact llama-server.  Mode 1 is a
numerical diagnostic; mode 2 must match mode 0 exactly before it is timed.
The measured time is synchronized ggml graph time, not isolated kernel time.
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
import shutil
import subprocess
import sys
import tempfile
import time


SHAPES = {(1024, 2048), (1024, 256), (2048, 1024), (1024, 4096), (4096, 1024)}
CASES = {"normal", "cancellation", "bounded"}
TARGETS = ("test-flashrt-gemm-bench", "test-flashrt-gemm-guard", "llama-server")
MODE_NAMES = {0: "original_ggml", 1: "cublaslt_diagnostic", 2: "ggml_mma16_candidate"}


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_json(path, value):
    path = Path(path)
    pending = path.with_name(path.name + ".tmp")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def execute(command, log, env=None, timeout=1800, acceptable=(0,)):
    """Only this Popen object's process can be terminated on failure/timeout."""
    command = [str(item) for item in command]
    print("运行:", " ".join(command), flush=True)
    started = time.monotonic()
    with Path(log).open("w") as output:
        child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, env=env)
        try:
            next_update = started + 10
            while child.poll() is None:
                now = time.monotonic()
                need(now - started < timeout, f"命令超时；日志 {log}")
                if now >= next_update:
                    print(f"处理中：{int(now - started)}s；日志 {Path(log).name}", flush=True)
                    next_update = now + 10
                time.sleep(0.1)
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
    text = Path(log).read_text(errors="replace")
    need(child.returncode in acceptable,
         f"命令退出 {child.returncode}；日志 {log}\n{text[-5000:]}")
    return child.returncode, text


def gpu_snapshot(gpu, output):
    nvidia = shutil.which("nvidia-smi")
    need(nvidia, "找不到 nvidia-smi；没有启动 GPU 测试。")
    query = [nvidia, "-i", str(gpu),
             "--query-gpu=index,uuid,name,compute_cap,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"]
    _, text = execute(query, output / "gpu.log", timeout=20)
    rows = list(csv.reader(text.strip().splitlines(), skipinitialspace=True))
    need(len(rows) == 1 and len(rows[0]) == 6, f"GPU 查询格式异常：{text}")
    index, uuid, name, capability, memory, utilization = (s.strip() for s in rows[0])
    need(capability == "8.9", f"GPU {index} 的 compute capability 为 {capability}，本阶段只验收 SM89。")
    try:
        used_mib, busy_pct = float(memory), float(utilization)
    except ValueError as error:
        raise RuntimeError("GPU 显存/利用率不可读，不能确认测试条件。") from error
    need(math.isfinite(used_mib) and math.isfinite(busy_pct), "GPU 查询返回非有限数值。")
    _, apps_text = execute([nvidia, "-i", str(gpu),
        "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits"], output / "gpu-processes.log", timeout=20)
    app_rows = list(csv.reader(apps_text.strip().splitlines(), skipinitialspace=True))
    need(all(len(row) == 4 and row[0].strip().startswith("GPU-") for row in app_rows),
         f"GPU 进程查询格式异常，不能确认设备空闲：{apps_text}")
    apps = [row for row in app_rows if row[0].strip() == uuid]
    snapshot = {"index": index, "uuid": uuid, "name": name, "compute_capability": capability,
                "memory_used_mib": used_mib, "utilization_pct": busy_pct, "compute_processes": apps}
    write_json(output / "gpu.json", snapshot)
    need(not apps and used_mib <= 256 and busy_pct <= 5,
         f"GPU {index} 当前有工作负载（显存 {used_mib:g} MiB，利用率 {busy_pct:g}%，"
         f"计算进程 {len(apps)} 个）。未停止任何已有进程；请选择空闲 GPU 后重新运行。")
    return snapshot


def source_hashes(root):
    paths = {root / "CMakeLists.txt"}
    suffixes = {".c", ".cc", ".cpp", ".cu", ".h", ".hpp", ".cuh", ".cmake", ".py"}
    helper = root / "run-ggml-math-shadow.py"
    if helper.is_file():
        paths.add(helper)
    for directory in ("ggml", "tests", "src", "common", "tools", "cmake", "scripts"):
        for path in (root / directory).rglob("*"):
            if path.is_file() and (path.name == "CMakeLists.txt" or path.suffix in suffixes):
                paths.add(path)
    return {str(p.relative_to(root)): sha256(p) for p in sorted(paths)}


def artifact_hashes(build):
    paths = {build / "bin" / name for name in TARGETS}
    paths.update(p for p in (build / "bin").glob("lib*.so*") if p.is_file())
    need(all(path.is_file() for path in paths), "构建产物缺失；请去掉 --skip-build 重新构建。")
    need(any(path.name.startswith("libggml-cuda") for path in paths), "找不到 libggml-cuda；不能验收 CUDA 构建。")
    return {str(p.relative_to(build)): sha256(p) for p in sorted(paths)}


def prepare_build(args, report, output):
    root, build = args.root, args.build
    manifest = build / "flashrt-gemm-stage-build.json"
    def save_snapshot(status):
        snapshot = output / "build-manifest.json"
        shutil.copyfile(manifest, snapshot)
        report["build"] = {"status": status, "directory": str(build), "manifest": str(snapshot),
                           "manifest_sha256": sha256(snapshot)}
    if args.skip_build:
        need(manifest.is_file(), "--skip-build 需要本脚本此前成功构建留下的清单；请先不带该参数运行一次。")
        previous = json.loads(manifest.read_text())
        need(previous.get("root") == str(root) and previous.get("sources") == source_hashes(root),
             "源码或仓库路径已改变；拒绝复用旧构建，请去掉 --skip-build。")
        need(previous.get("artifacts") == artifact_hashes(build),
             "构建产物已改变；拒绝复用旧清单，请去掉 --skip-build。")
        need(previous.get("cache_sha256") == sha256(build / "CMakeCache.txt"),
             "CMake 配置已改变；请去掉 --skip-build。")
        save_snapshot("reused_verified_build")
        return
    cmake = shutil.which("cmake")
    need(cmake, "找不到 cmake。")
    before = source_hashes(root)
    execute([cmake, "-S", root, "-B", build, "-DGGML_CUDA=ON", "-DGGML_CUDA_FLASHRT=ON",
             "-DCMAKE_CUDA_ARCHITECTURES=89", "-DLLAMA_BUILD_TESTS=ON"], output / "configure.log")
    execute([cmake, "--build", build, "--target", *TARGETS, "-j", str(args.jobs)],
            output / "build.log", timeout=args.build_timeout)
    after = source_hashes(root)
    need(before == after, "构建期间源码发生变化；请重新运行，避免验证混合版本。")
    data = {"schema_version": 1, "root": str(root), "sources": after,
            "artifacts": artifact_hashes(build), "cache_sha256": sha256(build / "CMakeCache.txt")}
    write_json(manifest, data)
    save_snapshot("built")


def environment(gpu, mode, graphs, trace):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=gpu["uuid"], GGML_CUDA_FLASHRT="1",
               GGML_CUDA_FLASHRT_GEMM=str(mode), GGML_CUDA_FLASHRT_TRACE=str(int(trace)),
               GGML_CUDA_FLASHRT_SHADOW="0", GGML_CUDA_FLASHRT_GEMM_SHADOW="0",
               GGML_CUDA_FLASHRT_GEMM_GRAPH_TRACE=str(int(trace)),
               GGML_CUDA_DISABLE_FUSION="0")
    if graphs == "off":
        env["GGML_CUDA_DISABLE_GRAPHS"] = "1"
    else:
        env.pop("GGML_CUDA_DISABLE_GRAPHS", None)
    return env


def record_key(record):
    return record["k"], record["n"], record["m"], record["case"]


def load_result(prefix, mode, cases, graphs, check_only):
    data = json.loads(prefix.with_suffix(".json").read_text())
    need(data.get("schema_version") == 1 and data.get("mode") == mode,
         "bench JSON 的版本或 mode 不匹配。")
    need(data.get("binary_dtype") == "f32le", "bench binary 格式必须为 f32le。")
    need(data.get("timing_metric") == "synchronized_backend_graph_wall_us",
         "bench 计时口径与本阶段协议不符。")
    need(data.get("backend") == "CUDA0", "bench 未在唯一可见设备 CUDA0 上运行。")
    need(data.get("warmup") == 5 and data.get("repeat") == 80,
         "bench 的 warmup/repeat 与本阶段协议不符。")
    need(type(data.get("graphs_requested")) is bool and
         data["graphs_requested"] is (graphs == "on") and data.get("check_only") is check_only,
         "bench JSON 的 graph/check_only 配置与请求不符。")
    records = data.get("records")
    need(isinstance(records, list), "bench JSON 缺少 records。")
    expected = {(k, n, 11, case) for k, n in SHAPES for case in cases}
    actual = [record_key(r) for r in records]
    need(len(actual) == len(set(actual)) and set(actual) == expected,
         "bench 缺少预期 shape/case，或存在重复记录。")
    raw = prefix.with_suffix(".bin").read_bytes()
    cursor = 0
    values = {}
    for record in records:
        key = record_key(record)
        count = record["elements"]
        need(type(count) is int and count == record["n"] * record["m"] and
             record["offset_elements"] == cursor, "binary offset/elements 不符合紧密顺序布局。")
        segment = raw[cursor * 4:(cursor + count) * 4]
        need(len(segment) == count * 4, "binary 数据截断。")
        decoded = array.array("f")
        decoded.frombytes(segment)
        if sys.byteorder != "little":
            decoded.byteswap()
        need(decoded.itemsize == 4 and all(math.isfinite(x) for x in decoded),
             f"{key}: 输出包含非有限值或主机 F32 格式不受支持。")
        for name in ("max_abs_fp64", "rmse_fp64", "reference_scale", "repeat_max_abs"):
            value = record.get(name)
            need(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                 f"{key}: {name} 不是有效误差/尺度。")
        need(type(record.get("fp64_bound_pass")) is bool, f"{key}: 缺少 FP64 检查状态。")
        if not check_only:
            samples = record.get("measurements_us")
            need(isinstance(samples, list) and len(samples) == 80 and all(
                type(value) in (int, float) and math.isfinite(value) and value > 0 for value in samples),
                f"{key}: 缺少 80 个有效原始计时样本。")
            for name in ("p50_us", "p95_us"):
                value = record.get(name)
                need(type(value) in (int, float) and math.isfinite(value) and value > 0,
                     f"{key}: {name} 不是有效耗时。")
            need(record["p95_us"] >= record["p50_us"], f"{key}: P95 小于 P50。")
            ordered = sorted(samples)
            for name, fraction in (("p50_us", 0.50), ("p95_us", 0.95)):
                position = fraction * (len(ordered) - 1)
                low = int(position)
                expected_quantile = ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (position - low)
                need(math.isclose(record[name], expected_quantile, rel_tol=1e-12, abs_tol=1e-9),
                     f"{key}: {name} 与原始计时样本不一致。")
        values[key] = (segment, decoded)
        cursor += count
    need(len(raw) == cursor * 4, "binary 末尾存在未登记数据。")
    return data, values


def compare(left, right):
    need(set(left) == set(right), "比较两组的 shape/case 不匹配。")
    rows = []
    for key in sorted(left):
        a_bytes, a = left[key]
        b_bytes, b = right[key]
        differences = [float(u) - float(v) for u, v in zip(a, b)]
        rows.append({"k": key[0], "n": key[1], "m": key[2], "case": key[3],
                     "max_abs": max(abs(d) for d in differences),
                     "rmse": math.sqrt(math.fsum(d * d for d in differences) / len(differences)),
                     "binary_equal": a_bytes == b_bytes})
    return rows


def trace_evidence(log, data, mode, graphs):
    evidence = []
    mode_tag = "1" if mode == 1 else "ggml_mma16"
    for record in data["records"]:
        def same_record(line):
            fields = dict(re.findall(r"\b(node|M|N|K|mode|capture)=([^\s]+)", line))
            return fields.get("node") == record["node"] and all(
                fields.get(big) == str(record[small]) for big, small in (("M", "m"), ("N", "n"), ("K", "k")))
        lines = [line for line in log.splitlines() if same_record(line)]
        hits = [line for line in lines if "flashrt_gemm:" in line and
                re.search(r"\bmode=" + re.escape(mode_tag) + r"(?:\s|$)", line)] if mode else []
        captured = [line for line in hits if re.search(r"\bcapture=1(?:\s|$)", line)]
        replays = sum("flashrt_gemm_graph: launch " in line for line in lines)
        evidence.append({"node": record["node"], "k": record["k"], "n": record["n"],
                         "case": record["case"], "candidate_host_enqueues": len(hits),
                         "candidate_capture_enqueues": len(captured), "actual_graph_launches": replays})
    if mode == 0:
        need("flashrt_gemm:" not in log, "mode 0 日志出现候选 GEMM 命中，基线不纯。")
    if graphs == "off":
        need(all(row["actual_graph_launches"] == 0 for row in evidence), "graphs-off 出现 graph launch。")
        need(all(row["candidate_capture_enqueues"] == 0 for row in evidence), "graphs-off 出现候选 capture。")
    return evidence


def benchmark(args, output, gpu, label, mode, graphs="off", check_only=True, cases=CASES, trace=True):
    prefix = output / label
    command = [args.build / "bin" / TARGETS[0], "--output-prefix", prefix, "--mode", str(mode),
               "--warmup", "5", "--repeat", "80", "--graphs", graphs,
               "--case", "normal" if cases == {"normal"} else "all"]
    if check_only:
        command.append("--check-only")
    code, text = execute(command, prefix.with_suffix(".log"),
                         env=environment(gpu, mode, graphs, trace), timeout=900, acceptable=(0, 2))
    data, values = load_result(prefix, mode, cases, graphs, check_only)
    result = {"label": label, "mode": mode, "graphs": graphs, "exit_code": code,
              "json": str(prefix.with_suffix(".json")), "binary": str(prefix.with_suffix(".bin")),
              "log": str(prefix.with_suffix(".log")), "records": data["records"]}
    if trace:
        result["trace"] = trace_evidence(text, data, mode, graphs)
    return result, values


def require_numerical(result):
    need(result["exit_code"] == 0 and all(r["fp64_bound_pass"] and r["repeat_max_abs"] == 0
                                         for r in result["records"]),
         f"{result['label']}: FP64 边界/重复运行稳定性检查失败；详见原始结果。")


def require_candidate(result):
    need(all(row["candidate_host_enqueues"] > 0 for row in result["trace"]),
         f"{result['label']}: 有 shape/case 未命中候选，不能把 fallback 算作验收。")


def check_graphs(args, output, gpu, report, mode, own_reference, baseline_reference):
    result, values = benchmark(args, output, gpu, f"graphs-on-mode-{mode}", mode, graphs="on")
    report["stages"].append(result)
    require_numerical(result)
    comparison = compare(own_reference, values)
    result["vs_own_graphs_off"] = comparison
    result["vs_mode_0_graphs_off"] = compare(baseline_reference, values)
    need(all(row["binary_equal"] for row in comparison),
         f"mode {mode}: graph-on 结果与该模式 graph-off 结果不一致。")
    if mode:
        require_candidate(result)
        need(all(row["candidate_capture_enqueues"] > 0 for row in result["trace"]),
             f"mode {mode}: 有候选未进入 CUDA Graph capture，不能验收其 graph replay。")
    need(all(row["actual_graph_launches"] >= 6 for row in result["trace"]),
         f"mode {mode}: 有 shape/case 未记录至少 6 次真实 cudaGraphLaunch。")
    write_json(output / "summary.json", report)
    return result


def measure_abba(args, output, gpu, report, candidate_mode, baseline_reference, candidate_reference):
    blocks = []
    normal = lambda values: {key: value for key, value in values.items() if key[3] == "normal"}
    reference_normal = normal(baseline_reference)
    candidate_normal = normal(candidate_reference)
    for block, mode in enumerate((0, candidate_mode, candidate_mode, 0), 1):
        label = f"perf-candidate-{candidate_mode}-block-{block}-mode-{mode}"
        result, values = benchmark(args, output, gpu, label, mode, graphs="on",
                                   check_only=False, cases={"normal"}, trace=False)
        report["stages"].append(result)
        require_numerical(result)
        result["vs_mode_0"] = compare(reference_normal, values)
        result["vs_own_correctness_result"] = compare(candidate_normal if mode else reference_normal, values)
        need(all(row["binary_equal"] for row in result["vs_own_correctness_result"]),
             f"mode {mode}: 性能阶段输出与该模式准确性结果不再逐位一致；保留记录并停止验收。")
        blocks.append(result)
        write_json(output / "summary.json", report)
    timing = []
    for k, n in sorted(SHAPES):
        sides = {}
        for mode in (0, candidate_mode):
            selected = [next(r for r in block["records"] if r["k"] == k and r["n"] == n)
                        for block in blocks if block["mode"] == mode]
            sides[mode] = {"block_p50_us": [r["p50_us"] for r in selected],
                           "block_p95_us": [r["p95_us"] for r in selected],
                           "mean_block_p50_us": math.fsum(r["p50_us"] for r in selected) / len(selected)}
        a, b = sides[0]["mean_block_p50_us"], sides[candidate_mode]["mean_block_p50_us"]
        timing.append({"k": k, "n": n, "m": 11, "baseline": sides[0], "candidate": sides[candidate_mode],
                       "mean_block_p50_latency_reduction_pct": 100 * (a - b) / a})
    print(f"\nmode {candidate_mode}: K,N,M,A_mean_block_p50_us,B_mean_block_p50_us,latency_reduction_pct")
    for row in timing:
        print(f"{row['k']},{row['n']},11,{row['baseline']['mean_block_p50_us']:.3f},"
              f"{row['candidate']['mean_block_p50_us']:.3f},{row['mean_block_p50_latency_reduction_pct']:.2f}")
    return {"order": [0, candidate_mode, candidate_mode, 0], "graphs": "on", "warmup_per_block": 5,
            "premeasurement_sanity_calls_per_shape": 12,
            "measurements_per_shape_per_block": 80, "rows": timing,
            "metric": "synchronized ggml graph time; mean of two block medians",
            "automatic_promotion": False, "accepted_for_model": False}


def run(args, report, output):
    initial = output / "gpu-before-build"
    initial.mkdir()
    report["gpu"] = gpu_snapshot(args.gpu, initial)
    prepare_build(args, report, output)
    ready = output / "gpu-before-tests"
    ready.mkdir()
    gpu = gpu_snapshot(args.gpu, ready)
    need(gpu["uuid"] == report["gpu"]["uuid"], "构建后 GPU 设备映射发生变化。")
    execute([args.build / "bin" / TARGETS[1]], output / "guard.log", timeout=60)
    report["guard"] = "passed"
    stages = report["stages"]
    references, checks = {}, {}
    mode2_error = None
    for mode in (0, 1, 2):
        try:
            result, values = benchmark(args, output, gpu, f"check-mode-{mode}", mode)
            stages.append(result)
        except Exception as error:
            if mode == 0:
                raise
            if mode == 2:
                mode2_error = f"{type(error).__name__}: {error}"
                report["mode2_correctness"] = {"status": "failed", "error": mode2_error}
                continue
            report["mode1_diagnostic"] = {"status": "failed", "accepted_for_model": False,
                                           "log": str(output / "check-mode-1.log"),
                                           "json": str(output / "check-mode-1.json"),
                                           "binary": str(output / "check-mode-1.bin"),
                                           "error": f"{type(error).__name__}: {error}"}
            continue
        references[mode], checks[mode] = values, result
        if mode == 0:
            require_numerical(result)
        else:
            comparison = compare(references[0], values)
            result["vs_mode_0"] = comparison
            if mode == 1:
                result["disposition"] = "diagnostic_only_not_accepted_or_enabled"
                result["all_shapes_hit"] = all(row["candidate_host_enqueues"] > 0 for row in result["trace"])
                print("mode 1：仅记录 cuBLASLt 数值诊断，不自动接入模型。", flush=True)
        write_json(output / "summary.json", report)

    baseline_graph = check_graphs(args, output, gpu, report, 0, references[0], references[0])
    if 1 in checks:
        diagnostic = report["mode1_diagnostic"] = {"status": "incomplete", "accepted_for_model": False}
        try:
            require_numerical(checks[1])
            require_candidate(checks[1])
            lt_graph = check_graphs(args, output, gpu, report, 1, references[1], references[0])
            diagnostic["graphs"] = {"same_as_own_graphs_off": True, "capture_and_replay_verified": True,
                "minimum_actual_graph_launches_per_case": min(r["actual_graph_launches"] for r in lt_graph["trace"])}
            diagnostic["performance"] = measure_abba(args, output, gpu, report, 1, references[0], references[1])
            diagnostic["status"] = "diagnostics_complete_not_accepted_for_model"
        except Exception as error:
            diagnostic["status"] = "failed"
            diagnostic["error"] = f"{type(error).__name__}: {error}"
            print("mode 1 诊断未通过；继续独立检查 mode 2：", diagnostic["error"], flush=True)
        write_json(output / "summary.json", report)

    need(mode2_error is None, f"mode 2 微测试未完成：{mode2_error}")
    require_numerical(checks[2])
    require_candidate(checks[2])
    need(all(row["binary_equal"] for row in checks[2]["vs_mode_0"]),
         "mode 2 与原 GGML 输出不是逐位一致；本次不运行 mode 2 性能测试。")
    report["mode_2_graphs_off"] = "binary_equal_for_five_shapes_three_cases"
    mma_graph = check_graphs(args, output, gpu, report, 2, references[2], references[0])
    report["mode2_correctness"] = {
        "status": "passed", "binary_equal": True, "replay_verified": True,
        "minimum_actual_graph_launches_per_case": min(
            row["actual_graph_launches"] for stage in (baseline_graph, mma_graph) for row in stage["trace"]),
        "shapes": [{"k": k, "n": n, "m": 11} for k, n in sorted(SHAPES)],
        "cases": sorted(CASES),
    }
    report["performance"] = measure_abba(args, output, gpu, report, 2, references[0], references[2])
    manifest = json.loads(Path(report["build"]["manifest"]).read_text())
    need(manifest["sources"] == source_hashes(args.root) and manifest["artifacts"] == artifact_hashes(args.build),
         "测试期间源码或二进制发生变化；拒绝把混合版本记录标记为通过。")
    report["status"] = "microtests_complete"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--build", type=Path)
    parser.add_argument("--gpu", default="2", help="nvidia-smi GPU index or UUID; default: 2")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--build-timeout", type=int, default=3600)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--summary-pointer", type=Path,
                        help="write a JSON object containing this run's exact summary path and status")
    args = parser.parse_args()
    args.root = args.root.expanduser().resolve()
    args.build = (args.build or args.root / "build-flashrt-sm89").expanduser().resolve()
    need(args.jobs > 0 and args.build_timeout > 0, "jobs/build-timeout 必须为正数。")
    result_root = args.root / "results" / "flashrt-sm89"
    result_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="gemm-stage-", dir=result_root))
    report = {"schema_version": 1, "status": "incomplete", "full_port_complete": False, "root": str(args.root),
              "build_directory": str(args.build), "modes": MODE_NAMES, "stages": [],
              "scope": "SM89 M=11 F16-weight/F32-input-output GEMM microtests only",
              "limitations": ["没有运行 pi0 模型端到端验收，未自动启用候选。",
                              "同步 ggml graph 耗时包含调度/同步成本；不是孤立 kernel 时间。",
                              "已记录 kernel 累计时间占比不能直接当作端到端提速幅度。",
                              "共享 GPU 运行期间可能新增负载；ABBA 原始区块结果完整保留。"]}
    print("结果目录:", output, flush=True)
    write_json(output / "summary.json", report)
    code = 0
    try:
        run(args, report, output)
    except (Exception, KeyboardInterrupt) as error:
        code = 1
        report["status"] = "incomplete"
        report["error"] = f"{type(error).__name__}: {error}"
        print("阶段未完成:", report["error"], file=sys.stderr, flush=True)
    finally:
        write_json(output / "summary.json", report)
        if args.summary_pointer:
            write_json(args.summary_pointer.expanduser().resolve(),
                       {"summary": str(output / "summary.json"), "status": report["status"]})
        print("status:", report["status"], flush=True)
        print("已保存:", output / "summary.json", flush=True)
        print("本阶段不代表完整 kernel 移植或模型端到端验收完成。", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
