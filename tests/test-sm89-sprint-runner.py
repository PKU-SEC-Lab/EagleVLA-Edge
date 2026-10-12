"""CPU-only orchestration checks. Fake outputs never serve as GPU evidence."""
import contextlib
import copy
import hashlib
import io
import json
import math
import os
import runpy
import subprocess
import struct
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPRINT = runpy.run_path(str(ROOT / "scripts/run-sm89-sprint.py"))
PROFILE = runpy.run_path(str(ROOT / "scripts/run-sm89-profile.py"))


def gguf_fixture():
    def string(value):
        data = value.encode()
        return struct.pack("<Q", len(data)) + data
    return (b"GGUF" + struct.pack("<IQQ", 3, 1, 1) + string("tokenizer.ggml.tokens") +
            struct.pack("<IIQ", 9, 8, 2) + string("alpha") + string("beta") +
            string("blk.0.attn_q.weight") + struct.pack("<IQQIQ", 2, 2048, 2048, 1, 0))


class RunnerTests(unittest.TestCase):
    def scenario(self, failure=None):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            root = home / "Jetson-PI-Edge-flashrt-sm89"
            results = root / "results/flashrt-sm89"
            results.mkdir(parents=True)
            binary = root / "build-flashrt-sm89/bin/llama-server"
            binary.parent.mkdir(parents=True)
            binary.touch()
            (binary.parent.parent / "CMakeCache.txt").write_text("mock")
            source = root / "ggml/src/ggml-cuda/flashrt-rmsnorm.cu"
            source.parent.mkdir(parents=True)
            source.write_text("mock")
            models = home / "models/Jetson-PI-GGUF/pi0"
            models.mkdir(parents=True)
            for name in ("pi_llm.gguf", "mmproj-model-f16.gguf"):
                (models / name).write_bytes(gguf_fixture())
            image = results / "image.png"
            image.write_bytes(b"mock image")
            def action(value=0):
                return [[float(value)] * 32 for _ in range(10)]
            reference = {"action": action(), "action_final_raw": action(), "action_final": action()}
            baseline = results / "baseline.json"
            baseline.write_text(json.dumps({"records": [{"response": reference}]}))
            proof = results / "ggml-math-terminal-9f94pkb6/summary.json"
            proof.parent.mkdir()
            proof.write_text(json.dumps({"status": "equal_in_both_settings", "command": [str(binary), "-lv", "5"],
                                         "input": {"image_sha256": hashlib.sha256(image.read_bytes()).hexdigest()}}))
            active = [None]
            children = []
            profile_calls = []
            state = ["0," * 31 + "0"]
            class Child:
                def __init__(self, args, **kwargs):
                    self.pid = 100 + len(children)
                    self.env = kwargs["env"]
                    self.alive = True
                    children.append(self)
                    active[0] = self
                    log = kwargs["stdout"]
                    if self.env["GGML_CUDA_FLASHRT_SHADOW"] == "1":
                        for dim in (1024, 2048):
                            err = 1e-4 if failure == "shadow" else 0
                            log.write(f"flashrt_rmsnorm_shadow: math=ggml node=n rows=11 dim={dim} nonfinite=0 "
                                      f"ab_max={err} ab_rmse=0 ab_rel_l2=0 propagate=ggml\n")
                    if self.env["GGML_CUDA_FLASHRT"] == "1" and self.env["GGML_CUDA_FLASHRT_TRACE"] == "1":
                        for dim in (1024, 2048):
                            log.write(f"flashrt_rmsnorm: enqueued rows=11 dim={dim} node=n math=ggml\n")
                    if "GGML_CUDA_DISABLE_GRAPHS" not in self.env:
                        log.write("CUDA graph warmup complete\n")
                    log.flush()
                def poll(self):
                    return None if self.alive else 0
            def stop(child):
                child.alive = False
                active[0] = None
            def flat(response, key):
                rows = response[key]
                if len(rows) != 10 or any(len(row) != 32 for row in rows):
                    raise ValueError("shape")
                values = [v for row in rows for v in row]
                if not all(math.isfinite(v) for v in values):
                    raise ValueError("nonfinite")
                return values
            def difference(a, b, key):
                d = [x-y for x,y in zip(flat(a,key), flat(b,key))]
                return {"max_abs": max(map(abs,d)), "rmse": math.sqrt(sum(x*x for x in d)/len(d))}
            def request(opener, method, route, payload=None, **kwargs):
                if failure == "http" and route == "/foreground/infer":
                    raise OSError("mock HTTP failure")
                if route.endswith("reset"):
                    return {"reset": True}
                if route.endswith("image"):
                    return {"loaded": True}
                if route.endswith("state"):
                    state[0] = payload["state"]
                    return {"loaded": True, "state_dim": 32}
                if route.endswith("infer"):
                    changed = payload["text"] != "fixed prompt" or state[0].startswith("0.1")
                    r = copy.deepcopy(reference)
                    if changed:
                        r["action_final_raw"] = action(.25)
                        r["action_final"] = action(.25)
                    c = active[0]
                    if failure == "action" and c.env["GGML_CUDA_FLASHRT"] == "1" and c.env["GGML_CUDA_FLASHRT_SHADOW"] == "0":
                        r["action_final_raw"][0][0] += .01
                    if failure == "nonfinite" and c.env["GGML_CUDA_FLASHRT"] == "1" and c.env["GGML_CUDA_FLASHRT_SHADOW"] == "0":
                        r["action_final_raw"][0][0] = float("nan")
                    r.update(vit_ms=6., encode_ms=14., decode_ms=34., total_ms=48., timing_breakdown_ms={"handler_total_ms": 55.})
                    return r
                return {"status": "ok"}
            def write_json(path, value):
                path.write_text(json.dumps(value, allow_nan=False))
            helper = dict(ROOT=root, RESULTS=results, MODEL_DIR=models, IMAGE=image, BASELINE=baseline,
                          MODE_ENV={"CUDA_VISIBLE_DEVICES": "2", "PI_MODEL": "pi0"}, PROMPT="fixed prompt",
                          flat_action=flat, difference=difference, write_json=write_json, request=request,
                          listener_pid=lambda: active[0].pid if active[0] else None,
                          stop_old_server=lambda *args: None, wait_ready=lambda *args: None, stop_child=stop)
            def command(args, path, cwd, env, **kwargs):
                text = "mock command\n"
                if str(args[0]) == "nvidia-smi":
                    text = "RTX 4090"
                if "test-backend-ops" in str(args[0]):
                    text = "CPU only, all skipped" if failure == "gpu-missing" else "RTX 4090 25/25 tests passed"
                    if env["GGML_CUDA_FLASHRT"] == "1":
                        text += " flashrt_rmsnorm: enqueued"
                path.write_text(text)
                return text
            def profile(*args, **kwargs):
                profile_calls.append(True)
                return {"status": "unavailable" if failure == "nsys-missing" else "captured"}
            g = SPRINT["main"].__globals__
            with patch.object(Path, "home", return_value=home), patch.object(g["runpy"], "run_path",
                    side_effect=[helper, {"find_nsys": lambda: "/mock/nsys", "profile": profile}]), \
                 patch.object(g["os"], "getpid", return_value=int(Path('/proc/self/stat').read_text().split(' ', 1)[0])), \
                 patch.object(g["shutil"], "which", return_value="/mock/cmake"), \
                 patch.object(g["subprocess"], "Popen", side_effect=Child), patch.dict(g, command_run=command), \
                 contextlib.redirect_stdout(io.StringIO()):
                if failure in ("action", "nonfinite", "shadow", "http", "gpu-missing"):
                    with self.assertRaises((RuntimeError, OSError, ValueError)):
                        SPRINT["main"]()
                else:
                    SPRINT["main"]()
            directory = next(results.glob("rmsnorm-sprint-*"))
            summary = json.loads((directory / "summary.json").read_text())
            self.assertFalse(any(c.alive for c in children), "owned server leaked")
            self.assertFalse(summary["full_port_complete"])
            if failure in ("action", "nonfinite", "shadow", "http", "gpu-missing"):
                self.assertEqual(summary["status"], "incomplete")
                self.assertFalse(profile_calls, "profile must not follow failed numerical gates")
                if failure == "action":
                    failed = json.loads((directory / "failed-response.json").read_text())
                    self.assertEqual(failed["actual"]["response"]["action_final_raw"][0][0], .01)
                if failure == "nonfinite":
                    failed = json.loads((directory / "failed-response.json").read_text())
                    self.assertEqual(failed["actual"]["response"]["action_final_raw"][0][0], "NaN")
            else:
                self.assertEqual(len(summary["runs"]), 4)
                self.assertEqual(sum(len(r["samples"]) for r in summary["runs"]), 80)
                self.assertEqual(summary["status"], "profile_pending" if failure else "rmsnorm_stage_complete")
                self.assertEqual(len(profile_calls), 1)

    def test_complete_orchestration(self): self.scenario()
    def test_action_mismatch_saves_evidence_and_stops(self): self.scenario("action")
    def test_nonfinite_saves_evidence_and_stops(self): self.scenario("nonfinite")
    def test_shadow_mismatch_stops(self): self.scenario("shadow")
    def test_http_failure_stops_owned_server(self): self.scenario("http")
    def test_cpu_only_test_is_not_gpu_success(self): self.scenario("gpu-missing")
    def test_missing_nsys_is_reported_as_pending(self): self.scenario("nsys-missing")
    def test_empty_profile_rejected(self):
        self.assertEqual(PROFILE["kernel_rows"]("SKIPPED: no CUDA kernel data"), [])
        data = '"Time (%)","Total Time (ns)","Instances","Name"\n100,12345,10,"kernel<A, B>"\n'
        self.assertEqual(PROFILE["kernel_rows"](data)[0]["Name"], "kernel<A, B>")

    def test_gguf_inventory_skips_token_array_and_reads_real_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.gguf"
            path.write_bytes(gguf_fixture())
            inventory = SPRINT["gguf_inventory"](path)
            self.assertEqual(inventory["dtype_counts"], {"F16": 1})
            self.assertEqual(inventory["tensors"][0]["ne"], [2048, 2048])
            self.assertEqual(inventory["tensors"][0]["name"], "blk.0.attn_q.weight")

    def test_gguf_truncated_header_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.gguf"
            path.write_bytes(gguf_fixture()[:-3])
            with self.assertRaisesRegex(RuntimeError, "截断"):
                SPRINT["gguf_inventory"](path)


class ProfileCliTests(unittest.TestCase):
    # Model the split CLI reported by gpu4. Mentioning a flag in help prose
    # does not make it a supported argument to that command.
    HELP = {
        "launch": """  --session-new=<name>
      Name the session; configure --sample and --cpuctxsw using start.
  -t, --trace=<api>
      APIs to trace.
  --cuda-graph-trace=<granularity>
      Trace graph nodes.
  -w, --show-output=<bool>
      Show application output.
""",
        "start": """  --session=<name>
      Select this session.
  -o, --output=<file>
      Save the report.
  -s, --sample=<scope>
      CPU sampling; system-wide is not supported on all platforms.
  --cpuctxsw=<scope>
      CPU context switches.
""",
        "stop": "  --session=<name>\n      Stop this session.\n",
        "shutdown": "  --session=<name>\n      Select this session.\n  --kill=<signal>\n      Signal policy.\n",
        "stats": "  -r, --report=<report>\n      Report name.\n  -f, --format=<format>\n      Output format.\n"
                 "  --force-export=<bool>\n      Regenerate the SQLite export.\n",
    }

    def check(self, helps):
        seen = []
        def control(nsys, args, out, env, **kwargs):
            seen.append(args[0])
            text = "mock nsys split CLI" if args == ["--version"] else helps[args[0]]
            Path(out).write_text(text)
            return text, 0
        with tempfile.TemporaryDirectory() as tmp, patch.dict(PROFILE["check_cli"].__globals__, control=control):
            PROFILE["check_cli"]("/mock/nsys", Path(tmp), {})
        return seen

    def test_each_command_uses_its_own_option_definitions(self):
        self.assertEqual(self.check(self.HELP), ["--version", "launch", "start", "stop", "shutdown", "stats"])

    def test_prose_reference_is_not_an_option_definition(self):
        helps = dict(self.HELP, start=self.HELP["start"].replace(
            "  --cpuctxsw=<scope>\n      CPU context switches.\n", "      See --cpuctxsw in profile help.\n"))
        with self.assertRaisesRegex(RuntimeError, "start.*--cpuctxsw"):
            self.check(helps)

    def test_deprecated_definition_is_not_supported(self):
        helps = dict(self.HELP, start=self.HELP["start"].replace(
            "CPU context switches.", "Deprecated. This option is not supported by the start command."))
        with self.assertRaisesRegex(RuntimeError, "start.*--cpuctxsw"):
            self.check(helps)

    def test_option_prefix_is_not_an_exact_match(self):
        helps = dict(self.HELP, stop="  --session-new=<name>\n      Use --session on other commands.\n")
        with self.assertRaisesRegex(RuntimeError, "stop.*--session"):
            self.check(helps)

    def test_stats_requires_force_export(self):
        helps = dict(self.HELP, stats=self.HELP["stats"].replace(
            "  --force-export=<bool>\n      Regenerate the SQLite export.\n", ""))
        with self.assertRaisesRegex(RuntimeError, "stats.*--force-export"):
            self.check(helps)

    def test_split_cli_completes_warm_capture_without_real_processes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "capture"
            active, calls, controls, closed, signals = [False], [], [], [], []
            pid, fd = 123456, 123457
            def popen(args, **kwargs):
                self.assertEqual(args[:2], ["/mock/nsys", "launch"])
                options = args[2:args.index(sys.executable)]
                for option in options:
                    self.assertIn(option.split("=", 1)[0],
                                  {"--session-new", "--trace", "--cuda-graph-trace", "--show-output"})
                (directory / "server-pid.json").write_text("mock identity")
                active[0] = True
                return types.SimpleNamespace(poll=lambda: None, wait=lambda **kw: 0)
            def control(nsys, args, out, env, **kwargs):
                controls.append(args)
                text = "ok"
                if args == ["--version"]:
                    text = "mock nsys split CLI"
                elif args[-1] == "--help":
                    text = self.HELP[args[0]]
                elif args[0] == "start":
                    self.assertEqual(len(calls), 5, "collection must start after warmup")
                    self.assertIn("--sample=none", args)
                    self.assertIn("--cpuctxsw=none", args)
                elif args[0] == "stop":
                    self.assertEqual(len(calls), 15)
                    (directory / "steady.nsys-rep").write_bytes(b"mock report, not CUDA evidence")
                elif args[0] == "stats":
                    self.assertIn("--force-export=true", args,
                                  "an older SQLite export must not block saved-trace statistics")
                    text = '"Total Time (ns)","Name"\n100,"mock kernel"\n'
                Path(out).write_text(text)
                return text, 0
            def infer():
                calls.append(len(calls))
                return {"mock": True}
            def send_signal(target, number):
                signals.append((target, number))
                active[0] = False
            fake_os = types.SimpleNamespace(pidfd_open=lambda _: fd, close=closed.append)
            fake_signal = types.SimpleNamespace(SIGTERM=15, pidfd_send_signal=send_signal)
            fake_select = types.SimpleNamespace(select=lambda *args: ([] if active[0] else [fd], [], []))
            fake_subprocess = types.SimpleNamespace(Popen=popen, DEVNULL=-3, STDOUT=-2,
                                                     TimeoutExpired=subprocess.TimeoutExpired)
            h = types.SimpleNamespace(ROOT=root, MODEL_DIR=root, listener_pid=lambda: pid if active[0] else None,
                                      verify_server_identity=lambda *args: None,
                                      request=lambda *args, **kwargs: {"status": "ok"},
                                      write_json=lambda path, data: path.write_text(json.dumps(data)))
            with patch.dict(PROFILE["profile"].__globals__, find_nsys=lambda: "/mock/nsys", control=control,
                            subprocess=fake_subprocess, os=fake_os, signal=fake_signal, select=fake_select,
                            owned_pidfd=lambda path: (pid, fd)), contextlib.redirect_stdout(io.StringIO()):
                result = PROFILE["profile"](h, ["/mock/llama-server"], {}, None, infer, directory)
            self.assertEqual(result["status"], "captured")
            self.assertEqual(len(result["records"]), 10)
            self.assertEqual(closed, [fd])
            self.assertEqual(signals, [(fd, 15)])
            self.assertIn("--cpuctxsw=none", result["start_command"])
            for args in controls:
                if args[0] in ("start", "stop", "shutdown") and args[-1] != "--help":
                    self.assertIn("--session=" + result["session"], args)
                if args[0] == "shutdown" and args[-1] != "--help":
                    self.assertIn("--kill=none", args)


@unittest.skipUnless(Path('/proc/self/stat').read_text().split(' ', 1)[0] == str(os.getpid()),
                     'procfs uses a different PID namespace; run these checks on gpu4')
class ProfileProcessTests(unittest.TestCase):
    """Exercise real child/PID cleanup with a fake profiler, without a GPU."""
    def scenario(self, failure=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "capture"
            owned = []
            launchers = []
            controls = []
            calls = []
            foreign = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
            real_popen = subprocess.Popen
            def popen(args, **kwargs):
                if args[0] == "/mock/nsys":
                    controls.append(args[1:])
                    # The profiler can exit successfully while its application
                    # remains running. Use the real wrapper and its PID file.
                    index = args.index("--show-output=true") + 1
                    child = real_popen(args[index:], **kwargs)
                    owned.append(child)
                    if failure == "launcher":
                        deadline = time.monotonic() + 3
                        while not (directory / "server-pid.json").exists():
                            if time.monotonic() > deadline:
                                raise RuntimeError("mock wrapper did not start")
                            time.sleep(.01)
                    if failure == "launcher-late":
                        options = dict(kwargs, stdin=subprocess.PIPE)
                        launcher = real_popen([sys.executable, "-c",
                                               "import sys; sys.stdin.buffer.read(1); raise SystemExit(1)"], **options)
                    else:
                        launcher = real_popen([sys.executable, "-c", "raise SystemExit(" +
                                               ("1" if failure == "launcher" else "0") + ")"], **kwargs)
                    launchers.append(launcher)
                    if failure == "launcher":
                        self.assertEqual(launcher.wait(timeout=3), 1)
                    return launcher
                return real_popen(args, **kwargs)
            def control(nsys, args, out, env, **kwargs):
                controls.append(args)
                text = "ok"
                if args[0] == "stop":
                    (directory / "steady.nsys-rep").write_bytes(b"mock report, not CUDA evidence")
                if args[0] == "stats":
                    text = '"Total Time (ns)","Name"\n100,"mock kernel"\n'
                    if failure == "empty":
                        text = "SKIPPED: no kernels"
                Path(out).write_text(text)
                return text, 0
            def listener():
                if owned and owned[0].poll() is None and (directory / "server-pid.json").exists():
                    return owned[0].pid
                return None
            def verify(pid, *args):
                self.assertEqual(pid, owned[0].pid)
                self.assertNotEqual(pid, foreign.pid)
            def infer():
                calls.append(1)
                if failure == "request" and len(calls) == 7:
                    raise OSError("mock inference failed during capture")
                if failure == "launcher-late" and len(calls) == 7:
                    launchers[0].communicate(input=b"x", timeout=3)
                    self.assertEqual(launchers[0].returncode, 1)
                return {"mock": True}
            h = types.SimpleNamespace(ROOT=root, MODEL_DIR=root, listener_pid=listener,
                                      verify_server_identity=verify, request=lambda *a, **kw: {"status": "ok"},
                                      write_json=lambda path, data: path.write_text(json.dumps(data)))
            g = PROFILE["profile"].__globals__
            try:
                with patch.dict(g, find_nsys=lambda: "/mock/nsys", check_cli=lambda *a: "mock version",
                                control=control), patch.object(g["subprocess"], "Popen", side_effect=popen), \
                     contextlib.redirect_stdout(io.StringIO()):
                    command = [sys.executable, "-c", "import time; time.sleep(60)"]
                    if failure:
                        with self.assertRaises((RuntimeError, OSError)):
                            PROFILE["profile"](h, command, os.environ, None, infer, directory)
                    else:
                        result = PROFILE["profile"](h, command, os.environ, None, infer, directory)
                        self.assertEqual(result["status"], "captured")
                        self.assertEqual(len(result["records"]), 10)
                summary = json.loads((directory / "summary.json").read_text())
                self.assertEqual(summary["status"], "incomplete" if failure else "captured")
                if failure == "launcher":
                    self.assertEqual(len(calls), 0)
                if failure == "launcher-late":
                    self.assertEqual(len(calls), 7)
                self.assertIsNone(foreign.poll(), "unrelated process must remain alive")
                for child in owned:
                    self.assertIsNotNone(child.wait(timeout=2), "owned server leaked")
                for args in controls:
                    if args[0] in ("start", "stop", "shutdown"):
                        self.assertIn("--session=" + summary["session"], args)
                    if args[0] == "shutdown":
                        self.assertIn("--kill=none", args)
            finally:
                for child in [*launchers, *owned, foreign]:
                    if child.poll() is None:
                        child.terminate()
                    child.wait(timeout=3)

    def test_warm_capture_and_owned_cleanup(self): self.scenario()
    def test_request_failure_stops_owned_process(self): self.scenario("request")
    def test_empty_report_stops_owned_process(self): self.scenario("empty")
    def test_launcher_failure_after_spawn_stops_owned_process(self): self.scenario("launcher")
    def test_launcher_failure_during_capture_stops_owned_process(self): self.scenario("launcher-late")


if __name__ == "__main__":
    unittest.main()
