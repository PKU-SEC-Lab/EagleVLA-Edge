#!/usr/bin/env python3
"""Read an existing bias/residual report and telemetry; never run or edit anything.

Usage: python3 inspect-bias-telemetry.py /path/to/summary.json
GPU timestamps lack a timezone and processes.csv lacks sample boundaries. They
are intentionally reported at block scope, never assigned to individual requests.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys


METRICS = ("vit_ms", "encode_ms", "decode_ms", "total_ms",
           "handler_total_ms", "infer_http_ms")


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def quantile(values, q):
    values = sorted(values)
    if not values:
        return None
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def fmt(value):
    return f"{value:.3f}" if finite(value) else "N/A"


def describe(values):
    values = [v for v in values if finite(v)]
    return (f"n={len(values)},min={fmt(min(values) if values else None)},"
            f"p50={fmt(quantile(values, .5))},p95={fmt(quantile(values, .95))},"
            f"max={fmt(max(values) if values else None)}")


def timing_fields(record):
    response = record.get("response", {})
    fields = {k: response[k] for k in METRICS[:4] if finite(response.get(k))}
    if finite(record.get("infer_http_ms")):
        fields["infer_http_ms"] = record["infer_http_ms"]
    detail = response.get("timing_breakdown_ms", {})
    if finite(detail.get("handler_total_ms")):
        fields["handler_total_ms"] = detail["handler_total_ms"]

    def descend(obj, prefix):
        if not isinstance(obj, dict):
            return
        for key, value in obj.items():
            name = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                descend(value, name)
            elif key.endswith("_ms") and finite(value):
                fields[name] = value

    descend(detail, "timing")
    return fields


def csv_rows(path):
    if not path.is_file():
        print("缺少文件:", path)
        return []
    with path.open(errors="replace", newline="") as stream:
        return [[s.strip() for s in row] for row in csv.reader(stream) if row]


def cpu_counters(sample):
    fields = sample.get("proc_stat", "").split()
    if len(fields) < 9 or fields[0] != "cpu":
        return None
    try:
        counters = [int(s) for s in fields[1:9]]
    except ValueError:
        return None
    # guest/guest_nice are already included in user/nice, so use first 8 only.
    return sum(counters), counters[3] + counters[4]


def psi_total(sample):
    for line in sample.get("cpu_pressure", "").splitlines():
        if line.startswith("some "):
            return number(dict(s.split("=", 1) for s in line.split()[1:]).get("total"))
    return None


def read_telemetry(stage, directory, gpu):
    rows = csv_rows(directory / "gpu.csv")
    valid = [r for r in rows if len(r) == 9 and r[2] == gpu]
    print(f"GPU: samples={len(valid)}, unparsed_or_other_gpu_rows={len(rows) - len(valid)}")
    if valid:
        print(f"GPU 原始本地时间范围: {valid[0][0]} .. {valid[-1][0]}（未指定时区，不转换 epoch）")
    for index, name in ((3, "memory_MiB"), (4, "gpu_util_pct"), (5, "memory_util_pct"),
                        (6, "power_W"), (7, "sm_clock_MHz"), (8, "memory_clock_MHz")):
        print("GPU", name, describe([number(r[index]) for r in valid]))
    apps = csv_rows(directory / "processes.csv")
    owned = str(stage.get("pid"))
    observed = sorted({(r[1], r[2]) for r in apps if len(r) == 4 and r[0] == gpu})
    foreign = [(pid, name) for pid, name in observed if pid != owned]
    print("区块内观测计算进程:", json.dumps(observed, ensure_ascii=False), "服务 PID:", owned)
    print("区块内非本次服务 PID:", json.dumps(foreign, ensure_ascii=False))
    print("进程 CSV 无采样时间/边界；没有观测到其他 PID 也不能排除短暂外载或其他干扰。")
    path = directory / "cpu.jsonl"
    samples, bad = [], 0
    if path.is_file():
        for line in path.read_text(errors="replace").splitlines():
            try:
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError("not an object")
                samples.append(item)
            except (ValueError, TypeError):
                bad += 1
    print(f"CPU: samples={len(samples)}, invalid_lines={bad}, "
          f"logical_cpus={sorted({s.get('cpu_count') for s in samples if type(s.get('cpu_count')) is int})}")
    print("CPU loadavg_1min:", describe([s["loadavg"][0] for s in samples
          if isinstance(s.get("loadavg"), list) and s["loadavg"]]))
    busy, pressure = [], []
    for a, b in zip(samples, samples[1:]):
        ca, cb = cpu_counters(a), cpu_counters(b)
        if ca and cb:
            total, idle = cb[0] - ca[0], cb[1] - ca[1]
            if total > 0 and 0 <= idle <= total:
                busy.append(100 * (1 - idle / total))
        pa, pb = psi_total(a), psi_total(b)
        ma, mb = a.get("monotonic_ns"), b.get("monotonic_ns")
        if all(finite(x) for x in (pa, pb, ma, mb)) and mb > ma and pb >= pa:
            pressure.append(100 * (pb - pa) / ((mb - ma) / 1000))
    print("整机 CPU busy_pct（含 iowait 视作 idle）:", describe(busy))
    print("整机 CPU PSI some_pct（采样增量）:", describe(pressure))
    print("以上覆盖 warmup 和 measure 全区块；1 秒采样只能提供背景，不能证明单请求卡顿原因。")


def inspect(path):
    report = json.loads(path.read_text())
    print("只读报告:", path)
    for key in ("status", "correctness", "fixture_correctness", "host_contract_checks",
                "correctness_requests", "full_port_complete", "cleanup_errors"):
        print(f"{key}: {report.get(key)}")
    print("gpu:", report.get("gpu", {}).get("uuid"))
    stages = report.get("stages", {})
    for label, stage in stages.items():
        if label.startswith("e2e-"):
            print(label, "status=", stage.get("status"), "patterns=", stage.get("pattern_counts"),
                  "distinct_nodes=", len(stage.get("distinct_nodes", [])),
                  "captures=", stage.get("captured_dispatches"),
                  "warmup=", stage.get("graph_warmup_observed"))
    expected = ("perf-1-mode-0", "perf-2-mode-1", "perf-3-mode-1", "perf-4-mode-0")
    medians = {}
    for label in expected:
        stage = stages.get(label)
        if not isinstance(stage, dict):
            print("缺少性能区块:", label)
            continue
        records = [r for r in stage.get("records", []) if r.get("phase") == "measure"]
        fields = [timing_fields(r) for r in records]
        print(f"\n[{label}] status={stage.get('status')}, measure_count={len(records)}")
        print("environment:", json.dumps(stage.get("environment", {}), sort_keys=True))
        medians[label] = {}
        for metric in METRICS:
            values = [f[metric] for f in fields if metric in f]
            medians[label][metric] = quantile(values, .5)
            print(metric, describe(values))
        http = [f["infer_http_ms"] for f in fields if "infer_http_ms" in f]
        print(f"描述性慢样本计数: >100ms={sum(x > 100 for x in http)}/{len(http)}, "
              f">150ms={sum(x > 150 for x in http)}/{len(http)}；没有剔除样本。")
        print("HTTP 最慢 3 次（嵌套计时可能重叠，不能累加）:")
        ranked = sorted(zip(records, fields), key=lambda pair: pair[1].get("infer_http_ms", -1), reverse=True)
        for record, values in ranked[:3]:
            wall = record.get("started_wall_ns")
            utc = datetime.fromtimestamp(wall / 1e9, timezone.utc).isoformat() if finite(wall) else "N/A"
            print(f"request={record.get('request_index')}, start_UTC={utc}",
                  json.dumps({k: round(v, 3) for k, v in values.items()}, ensure_ascii=False, sort_keys=True))
        fast = [f for f in fields if f.get("infer_http_ms", math.inf) <= 100]
        slow = [f for f in fields if f.get("infer_http_ms", -1) > 100]
        print(f"HTTP<=100ms({len(fast)}) / >100ms({len(slow)}) 各指标 P50（描述性分组）:")
        for key in METRICS:
            print(key, fmt(quantile([f[key] for f in fast if key in f], .5)),
                  fmt(quantile([f[key] for f in slow if key in f], .5)))
        # Prefer report-adjacent path so copied report bundles still work.
        directory = path.parent / (label + "-telemetry")
        if not directory.is_dir():
            directory = Path(stage.get("telemetry", {}).get("directory", str(directory)))
        read_telemetry(stage, directory, report.get("gpu", {}).get("uuid"))
    print("\npaired_block_metric,A_p50_ms,B_p50_ms,latency_reduction_pct")
    for a, b in ((expected[0], expected[1]), (expected[3], expected[2])):
        for key in METRICS:
            av, bv = medians.get(a, {}).get(key), medians.get(b, {}).get(key)
            if finite(av) and finite(bv) and av > 0:
                print(f"{a}/{b}:{key},{av:.6f},{bv:.6f},{100 * (1 - bv / av):.4f}")
    print("已保存 pooled 指标:", json.dumps(report.get("performance", {}).get("metrics", {}),
                                          ensure_ascii=False, sort_keys=True))
    print("未启动服务、未修改报告、未改变验收门槛。区块遥测不能自动区分 GPU 外载、主机调度或其他延迟来源。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    try:
        inspect(args.summary.resolve(strict=True))
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"读取失败: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
