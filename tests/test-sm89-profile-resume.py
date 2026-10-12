#!/usr/bin/env python3
"""CPU-only orchestration regressions. These are not GPU/profile evidence."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

SOURCE = Path(__file__).resolve().parents[1] / "scripts/resume-sm89-profile.py"
spec = importlib.util.spec_from_file_location("profile_resume", SOURCE)
resume = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resume)
KEYS = ("action", "action_final_raw", "action_final")
METRICS = ("vit_ms", "encode_ms", "decode_ms", "total_ms", "handler_total_ms", "infer_http_ms")


def response():
    return {k: [[0.0] * 32 for _ in range(10)] for k in KEYS}


def flat_action(value, key):
    rows = value[key]
    if type(rows) is not list or len(rows) != 10:
        raise ValueError(f"{key}: shape")
    for row in rows:
        if type(row) is not list or len(row) != 32:
            raise ValueError(f"{key}: shape")
        for v in row:
            if type(v) not in (float, int) or not math.isfinite(v):
                raise ValueError(f"{key}: nonfinite/type")
    return [v for row in rows for v in row]


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.results = self.root / "results/flashrt-sm89"
        self.results.mkdir(parents=True)
        self.binary = self.root / "build-flashrt-sm89/bin/llama-server"
        self.binary.parent.mkdir(parents=True)
        self.binary.write_text("test fixture, not executable")
        self.image = self.results / "test.png"
        self.image.write_bytes(b"image fixture")
        self.proof = self.results / "proof.json"
        self.proof.write_text(json.dumps({"command": [str(self.binary), "-lv", "5"]}))
        self.source = self.results / "prior/summary.json"
        self.source.parent.mkdir()
        self.prior = {
            "correctness": "equal_for_three_input_cases_in_both_graph_settings",
            "rmsnorm_status": "validated_for_tested_inputs_and_benchmarked",
            "prior_proof": str(self.proof),
            "inputs": {"image_sha256": hashlib.sha256(self.image.read_bytes()).hexdigest(),
                       "cases": [{"id": c, "state": "0," * 31 + "0", "text": c}
                                 for c in ("fixed", "changed-text", "changed-state")]},
            "runs": [{"mode": m, "samples": [{k: 1.0 for k in METRICS} for _ in range(20)]}
                     for m in (0, 1, 1, 0)],
            "stages": {},
            "performance": {"metrics": {k: {"A_p50": 1.0, "B_p50": 0.9,
                                              "A_p95": 1.2, "B_p95": 1.1,
                                              "p50_latency_reduction_pct": 10.0}
                                         for k in METRICS}},
        }
        for graphs, sequence in ((0, ["fixed", "changed-text", "changed-state"]),
                                 (1, ["fixed"] * 6 + ["changed-text"] * 6 + ["changed-state"] * 6 + ["fixed"] * 4)):
            for mode in (0, 1):
                self.prior["stages"][f"e2e-graphs-{graphs}-mode-{mode}"] = {
                    "status": "requests_passed", "mode": mode, "graphs_allowed": bool(graphs),
                    "shadow": False, "rmsnorm_enqueues": 442 if mode else 0,
                    "graph_warmups": 3 if graphs else 0,
                    "records": [{"case": c, "response": response(), "infer_http_ms": 1.0} for c in sequence],
                }
        for i, mode in enumerate((0, 1, 1, 0), 1):
            self.prior["stages"][f"perf-{i}-mode-{mode}"] = {
                "status": "requests_passed", "mode": mode, "graphs_allowed": True, "shadow": False,
                "env": {"CUDA_VISIBLE_DEVICES": "2", "PI_MODEL": "pi0", "GGML_CUDA_FLASHRT": str(mode)},
                "records": [{"case": "fixed", "response": response(), "infer_http_ms": 1.0} for _ in range(20)],
            }
        self.profile_calls = []
        self.requests = []
        self.listener = None
        self.action_mutator = None
        self.profile_result_status = "captured"
        self.profile_exception = None
        self.profile_result_mutator = None
        self.output = io.StringIO()

        def difference(actual, expected, key):
            deltas = [a - b for a, b in zip(flat_action(actual, key), flat_action(expected, key))]
            return {"max_abs": max(map(abs, deltas)), "rmse": math.sqrt(sum(d*d for d in deltas)/len(deltas))}

        def write_json(path, data):
            path.write_text(json.dumps(data, allow_nan=False))

        def request(opener, method, route, payload=None, timeout=10):
            self.requests.append((method, route, payload))
            if route == "/foreground/reset":
                return {"reset": True}
            if route == "/foreground/image":
                return {"loaded": True}
            if route == "/foreground/state":
                return {"loaded": True, "state_dim": 32}
            if route == "/foreground/infer":
                value = response()
                if self.action_mutator:
                    self.action_mutator(value)
                return value
            raise AssertionError(route)

        self.h = dict(ROOT=self.root, RESULTS=self.results, IMAGE=self.image, request=request,
                      listener_pid=lambda: self.listener, flat_action=flat_action, difference=difference,
                      write_json=write_json)

        def profile(h, command, env, opener, infer, directory, mode=0):
            self.profile_calls.append((command, env, directory, mode))
            if self.profile_exception:
                raise self.profile_exception
            if self.profile_result_status == "unavailable":
                return {"status": "unavailable", "reason": "fixture nsys unavailable"}
            for _ in range(5):
                infer()
            records = [infer() for _ in range(10)]
            result = {"status": self.profile_result_status, "records": records,
                      "kernels": [{"Name": "fixture_kernel", "Total Time": "1"}]}
            if self.profile_result_mutator:
                self.profile_result_mutator(result)
            return result

        def modules(path):
            if Path(path).name == "run-ggml-math-shadow.py":
                return self.h
            if Path(path).name == "run-sm89-profile.py":
                return {"profile": profile}
            raise AssertionError(path)

        self.addCleanup(mock.patch.stopall)
        mock.patch.object(resume, "__file__", str(self.root / "scripts/resume-sm89-profile.py")).start()
        mock.patch.object(resume.runpy, "run_path", side_effect=modules).start()
        mock.patch("subprocess.run", side_effect=AssertionError("resume must not rebuild/run suite")).start()
        mock.patch("subprocess.Popen", side_effect=AssertionError("unexpected local process")).start()

    def save_prior(self):
        self.source.write_text(json.dumps(self.prior))
        self.source_bytes = self.source.read_bytes()

    def run_resume(self):
        self.save_prior()
        with contextlib.redirect_stdout(self.output):
            return resume.main(self.source)

    def latest_report(self):
        return json.loads(next(self.results.glob("rmsnorm-profile-resume-*/summary.json")).read_text())

    def test_success_profiles_only_and_preserves_original(self):
        self.assertEqual(self.run_resume(), 0)
        self.assertEqual(self.source.read_bytes(), self.source_bytes)
        self.assertEqual(len(self.profile_calls), 1)
        self.assertEqual(sum(route == "/foreground/infer" for _, route, _ in self.requests), 15)
        self.assertEqual(self.latest_report()["status"], "rmsnorm_stage_complete")
        self.assertFalse(self.latest_report()["full_port_complete"])
        args, env, directory, mode = self.profile_calls[0]
        self.assertEqual(args[-1], "3")
        self.assertEqual(mode, 0)
        self.assertEqual(env["GGML_CUDA_FLASHRT"], "0")
        self.assertNotIn("GGML_CUDA_DISABLE_GRAPHS", env)

    def test_foreign_listener_is_not_adopted(self):
        self.listener = 12345
        with self.assertRaisesRegex(RuntimeError, "18081"):
            self.run_resume()
        self.assertEqual(self.profile_calls, [])
        self.assertEqual(self.requests, [])
        self.assertEqual(self.source.read_bytes(), self.source_bytes)

    def test_cleanup_incomplete_cannot_become_success(self):
        self.profile_result_status = "incomplete"
        with self.assertRaisesRegex(RuntimeError, "进程清理"):
            self.run_resume()
        self.assertEqual(self.latest_report()["status"], "profile_incomplete")

    def test_profile_exception_preserved(self):
        self.profile_exception = RuntimeError("fixture nsys launch failed")
        with self.assertRaisesRegex(RuntimeError, "launch failed"):
            self.run_resume()
        self.assertEqual(self.latest_report()["status"], "profile_incomplete")
        self.assertIn("launch failed", self.latest_report()["error"])

    def test_unavailable_is_pending(self):
        self.profile_result_status = "unavailable"
        self.assertEqual(self.run_resume(), 2)
        self.assertEqual(self.latest_report()["status"], "profile_pending")
        self.assertEqual(self.requests, [])

    def invalid_action(self, value):
        self.action_mutator = lambda r: r["action_final_raw"][0].__setitem__(0, value)
        with self.assertRaises((ValueError, RuntimeError)):
            self.run_resume()
        self.assertEqual(self.latest_report()["status"], "profile_incomplete")
        path = next(self.results.glob("rmsnorm-profile-resume-*/failed-response.json"))
        evidence = json.loads(path.read_text(), parse_constant=lambda v: self.fail(f"non-JSON constant {v}"))
        self.assertEqual(evidence["actual"]["case"], "fixed")
        self.assertEqual(self.source.read_bytes(), self.source_bytes)
        return evidence["actual"]["response"]["action_final_raw"][0][0]

    def test_nan_retains_strict_json_evidence(self):
        self.assertEqual(self.invalid_action(float("nan")), "NaN")

    def test_null_retains_evidence(self):
        self.assertIsNone(self.invalid_action(None))

    def test_finite_difference_retains_evidence(self):
        self.assertEqual(self.invalid_action(0.25), 0.25)

    def test_wrong_graph_on_count_rejected_before_profile(self):
        self.prior["stages"]["e2e-graphs-1-mode-1"]["records"].pop()
        with self.assertRaisesRegex(RuntimeError, "端到端记录不完整"):
            self.run_resume()
        self.assertEqual(self.profile_calls, [])

    def test_missing_graph_warmup_rejected_before_profile(self):
        self.prior["stages"]["e2e-graphs-1-mode-1"]["graph_warmups"] = 0
        with self.assertRaisesRegex(RuntimeError, "证据缺失"):
            self.run_resume()
        self.assertEqual(self.profile_calls, [])

    def test_wrong_abba_order_rejected_before_profile(self):
        self.prior["runs"][3]["mode"] = 1
        with self.assertRaisesRegex(RuntimeError, "ABBA"):
            self.run_resume()
        self.assertEqual(self.profile_calls, [])

    def test_wrong_graph_input_sequence_rejected_before_profile(self):
        self.prior["stages"]["e2e-graphs-1-mode-1"]["records"][6]["case"] = "fixed"
        with self.assertRaisesRegex(RuntimeError, "输入切换顺序"):
            self.run_resume()
        self.assertEqual(self.profile_calls, [])

    def test_missing_abba_service_record_rejected_before_profile(self):
        self.prior["stages"]["perf-2-mode-1"]["records"].pop()
        with self.assertRaisesRegex(RuntimeError, "ABBA 服务记录"):
            self.run_resume()
        self.assertEqual(self.profile_calls, [])

    def test_unfinished_abba_service_rejected_before_profile(self):
        self.prior["stages"]["perf-4-mode-0"]["status"] = "running"
        with self.assertRaisesRegex(RuntimeError, "ABBA 服务记录"):
            self.run_resume()
        self.assertEqual(self.profile_calls, [])

    def test_new_binary_rejected_before_profile(self):
        self.save_prior()
        future = self.source.stat().st_mtime_ns + 10_000_000_000
        os.utime(self.binary, ns=(future, future))
        with contextlib.redirect_stdout(self.output), self.assertRaisesRegex(RuntimeError, "二进制或共享库已更新"):
            resume.main(self.source)
        self.assertEqual(self.profile_calls, [])

    def test_empty_kernel_summary_cannot_be_success(self):
        self.profile_result_mutator = lambda result: result.update(kernels=[])
        with self.assertRaisesRegex(RuntimeError, "采集数据"):
            self.run_resume()
        self.assertEqual(self.latest_report()["status"], "profile_incomplete")

    def test_short_capture_cannot_be_success(self):
        self.profile_result_mutator = lambda result: result["records"].pop()
        with self.assertRaisesRegex(RuntimeError, "采集数据"):
            self.run_resume()
        self.assertEqual(self.latest_report()["status"], "profile_incomplete")

    def test_new_library_rejected_before_profile(self):
        self.save_prior()
        library = self.binary.parent / "libfixture.so"
        library.write_text("changed library")
        future = self.source.stat().st_mtime_ns + 10_000_000_000
        os.utime(library, ns=(future, future))
        with contextlib.redirect_stdout(self.output), self.assertRaisesRegex(RuntimeError, "共享库已更新"):
            resume.main(self.source)
        self.assertEqual(self.profile_calls, [])


if __name__ == "__main__":
    unittest.main()
