#!/usr/bin/env python3
"""Complete only the warm profile after saved RMSNorm correctness and ABBA gates."""
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import sys
import tempfile
import time
import types
import urllib.request


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def checked_prior(path):
    raw = path.read_bytes()
    prior = json.loads(raw)
    need(prior.get("correctness") == "equal_for_three_input_cases_in_both_graph_settings",
         "原记录未通过三种输入的端到端检查，不能只补跑采集")
    need(prior.get("rmsnorm_status") == "validated_for_tested_inputs_and_benchmarked",
         "原记录尚未完成 RMSNorm 验收和测速")
    runs = prior.get("runs", [])
    need([r.get("mode") for r in runs] == [0, 1, 1, 0], "缺少完整 ABBA 四组结果")
    need(all(len(r.get("samples", [])) == 20 for r in runs), "ABBA 测量数不完整")
    for index, mode in enumerate((0, 1, 1, 0), 1):
        stage = prior["stages"][f"perf-{index}-mode-{mode}"]
        need(stage.get("status") == "requests_passed" and stage.get("mode") == mode
             and stage.get("graphs_allowed") is True and not stage.get("shadow")
             and [r.get("case") for r in stage.get("records", [])] == ["fixed"] * 20,
             "原 ABBA 服务记录不完整")
    for graphs, count in ((0, 3), (1, 22)):
        expected_cases = (["fixed"] * 6 + ["changed-text"] * 6 + ["changed-state"] * 6 + ["fixed"] * 4
                          if graphs else ["fixed", "changed-text", "changed-state"])
        for mode in (0, 1):
            stage = prior["stages"][f"e2e-graphs-{graphs}-mode-{mode}"]
            need(stage.get("status") == "requests_passed" and len(stage.get("records", [])) == count,
                 "原端到端记录不完整")
            need([r.get("case") for r in stage["records"]] == expected_cases, "原输入切换顺序不完整")
            need(stage.get("mode") == mode and stage.get("graphs_allowed") == bool(graphs)
                 and not stage.get("shadow"), "原端到端模式不符")
            need(bool(stage.get("rmsnorm_enqueues")) == bool(mode)
                 and bool(stage.get("graph_warmups")) == bool(graphs), "原融合或 CUDA Graph 证据缺失")
    performance = prior["performance"]["metrics"]
    print("复用已完成的三输入端到端检查和 80 次 ABBA 测量。", flush=True)
    print("metric,A_p50_ms,B_p50_ms,p50_latency_reduction_pct", flush=True)
    for key in ("vit_ms", "encode_ms", "decode_ms", "total_ms", "handler_total_ms", "infer_http_ms"):
        row = performance[key]
        values = [row[k] for k in ("A_p50", "B_p50", "p50_latency_reduction_pct")]
        need(all(type(v) in (int, float) and math.isfinite(v) for v in values), "原性能字段无效")
        print(f"{key},{values[0]:.3f},{values[1]:.3f},{values[2]:.2f}", flush=True)
    return prior, hashlib.sha256(raw).hexdigest()


def main(source):
    root = Path(__file__).resolve().parents[1]
    source = Path(source).resolve(strict=True)
    prior, prior_sha = checked_prior(source)
    h = types.SimpleNamespace(**runpy.run_path(str(root / "run-ggml-math-shadow.py")))
    p = types.SimpleNamespace(**runpy.run_path(str(root / "scripts/run-sm89-profile.py")))
    proof = json.loads(Path(prior["prior_proof"]).read_text())
    command = list(proof["command"])
    binary = (root / "build-flashrt-sm89/bin/llama-server").resolve(strict=True)
    need(command[0] == str(binary), "服务路径与原记录不一致")
    need(Path(h.ROOT).resolve() == root, "helper 指向了其他工作目录")
    need(hashlib.sha256(h.IMAGE.read_bytes()).hexdigest() == prior["inputs"]["image_sha256"],
         "图片与已验收输入不一致")
    # No build occurs here. Changed local executables/libraries need a fresh
    # correctness run, rather than attribution to an older successful run.
    artifacts = {binary} | {p.resolve() for p in binary.parent.glob("*.so*") if p.is_file()}
    need(all(p.stat().st_mtime_ns <= source.stat().st_mtime_ns for p in artifacts),
         "原记录之后二进制或共享库已更新，请重新做正确性验收")
    need(h.listener_pid() is None, "18081 被占用；本次未停止或接管任何现有服务")
    command[command.index("-lv") + 1] = "3"
    saved = prior["stages"]["perf-1-mode-0"]
    need(saved["env"].get("GGML_CUDA_FLASHRT") == "0", "原性能组不是 GGML 基线")
    env = dict(os.environ, **saved["env"])
    env.update(PI_MODEL_DEBUG="0", LLAMA_PI0_PERF="0")
    for key in ("PI0_ACTION_NOISE_BIN", "PI0_NORM_STATS_JSON", "PI_MODEL_DEBUG_DUMP_FILE",
                "PI05_DEBUG_DUMP_FILE", "GGML_CUDA_DISABLE_GRAPHS"):
        env.pop(key, None)
    need(env.get("CUDA_VISIBLE_DEVICES") == "2" and env.get("PI_MODEL") == "pi0", "原设备或模型设置不符")
    case = next(c for c in prior["inputs"]["cases"] if c["id"] == "fixed")
    reference = saved["records"][0]["response"]
    keys = ("action", "action_final_raw", "action_final")
    for key in keys:
        h.flat_action(reference, key)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    out = Path(tempfile.mkdtemp(prefix="rmsnorm-profile-resume-", dir=h.RESULTS))
    report = {"status": "running", "scope": "warm baseline profile; prior numerical and ABBA results retained",
              "full_port_complete": False, "source_summary": str(source), "source_summary_sha256": prior_sha,
              "correctness": prior["correctness"], "performance": prior["performance"]}
    print("本次采集结果目录:", out, flush=True)

    def infer():
        need(h.request(opener, "POST", "/foreground/reset").get("reset") is True, "reset 失败")
        need(h.request(opener, "POST", "/foreground/image", {"path": str(h.IMAGE)}, timeout=30).get("loaded") is True,
             "图片加载失败")
        state = h.request(opener, "PUT", "/foreground/state", {"state": case["state"]})
        need(state.get("loaded") is True and state.get("state_dim") == 32, "state 加载失败")
        start = time.perf_counter()
        response = h.request(opener, "POST", "/foreground/infer", {"text": case["text"]}, timeout=120)
        record = {"case": "fixed", "response": response, "infer_http_ms": (time.perf_counter()-start)*1000}
        try:
            for key in keys:
                h.flat_action(response, key)
                difference = h.difference(response, reference, key)
                need(difference["max_abs"] == 0, f"{key}: 与原基线不一致 {difference}")
        except BaseException:
            evidence = json.loads(json.dumps({"actual": record, "expected": reference}),
                                  parse_constant=lambda value: value)
            h.write_json(out / "failed-response.json", evidence)
            raise
        return record

    try:
        result = p.profile(h, command, env, opener, infer, out / "profile-baseline", mode=0)
        report["profile"] = result
        if result["status"] == "unavailable":
            report["status"] = "profile_pending"
            print(result["reason"], flush=True)
            return 2
        need(result["status"] == "captured" and len(result.get("records", [])) == 10
             and bool(result.get("kernels")), "采集数据或进程清理不完整")
        report["status"] = "rmsnorm_stage_complete"
        report["rmsnorm_status"] = prior["rmsnorm_status"]
        print("采集完成：10 次请求动作与原基线一致，CUDA kernel 数据非空。", flush=True)
        return 0
    except BaseException as exc:
        report.update(status="profile_incomplete", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        h.write_json(out / "summary.json", report)
        print("status:", report["status"], flush=True)
        print("correctness:", report["correctness"], flush=True)
        print("已保存:", out / "summary.json", flush=True)
        print("原验收记录保留；完整 kernel 移植仍未完成。", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("用法: resume-sm89-profile.py <已完成正确性及ABBA的summary.json>")
    raise SystemExit(main(sys.argv[1]))
