#!/usr/bin/env python3
"""Warm inference CUDA trace; no global profiler session or process selection."""
import csv
import glob
import io
import json
import os
import re
import select
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path


def start_ticks(pid):
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]


def server_exec(pidfile, argv):
    path = Path(pidfile)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"pid": os.getpid(), "ticks": start_ticks(os.getpid())}))
    tmp.replace(path)
    os.execv(argv[0], argv)


def owned_pidfd(pidfile):
    identity = json.loads(Path(pidfile).read_text())
    pid = int(identity["pid"])
    fd = os.pidfd_open(pid)
    try:
        if start_ticks(pid) != identity["ticks"] or Path(f"/proc/{pid}").stat().st_uid != os.getuid():
            raise RuntimeError("新启动服务的进程身份发生变化")
        return pid, fd
    except BaseException:
        os.close(fd)
        raise


def find_nsys():
    candidates = [shutil.which("nsys"), "/usr/local/bin/nsys"]
    for pattern in ("/opt/nvidia/nsight-systems/*/bin/nsys", "/opt/nvidia/nsight-systems/*/target-linux-x64/nsys",
                    "/usr/local/cuda*/bin/nsys", "/usr/local/cuda*/nsight-systems*/bin/nsys",
                    "/usr/local/cuda*/nsight-systems*/target-linux-x64/nsys"):
        candidates.extend(sorted(glob.glob(pattern), reverse=True))
    for path in candidates:
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return str(Path(path).resolve())
    return None


def control(nsys, args, out, env, *, timeout=180, required=True):
    command = [nsys, *args]
    with Path(out).open("w") as log:
        p = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                             stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + timeout
        while True:
            try:
                code = p.wait(timeout=min(10, max(.1, deadline - time.monotonic())))
                break
            except subprocess.TimeoutExpired:
                if time.monotonic() >= deadline:
                    p.terminate()
                    p.wait(timeout=10)
                    raise RuntimeError(f"nsys {args[0]} 超时；日志 {out}")
                print(f"nsys {args[0]} 正在处理…", flush=True)
    text = Path(out).read_text(errors="replace")
    if code and required:
        raise RuntimeError(f"nsys {args[0]} 失败 ({code}):\n{text[-4000:]}")
    return text, code


def cli_options(help_text):
    """Read option definitions, not option names quoted in help prose."""
    headers = list(re.finditer(
        r"^([ \t]*)(?:-[A-Za-z0-9?],?[ \t]+)?(--[a-z][a-z0-9-]*)(?=[=\s,]|$)",
        help_text, re.MULTILINE))
    if not headers:
        return set()
    # Descriptions can themselves start with another option name. They are
    # indented more deeply than the option declarations in nsys CLI help.
    indent = min(len(m[1].expandtabs()) for m in headers)
    headers = [m for m in headers if len(m[1].expandtabs()) == indent]
    supported = set()
    for i, header in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) else len(help_text)
        block = help_text[header.start():end]
        unavailable = (r"(?:^|\n)[ \t]*(?:Deprecated|Obsolete)\b|"
                       r"\b(?:This|The) (?:switch|option) is (?:deprecated|obsolete)\b|"
                       r"\bnot supported by (?:the )?\w+ command\b")
        if not (re.search(unavailable, block, re.IGNORECASE) or
                re.search(r"\b(?:deprecated|obsolete)\b", block.splitlines()[0], re.IGNORECASE)):
            supported.add(header[2])
    return supported


def check_cli(nsys, directory, env):
    version, _ = control(nsys, ["--version"], directory / "nsys-version.txt", env, timeout=15)
    for verb, options in {
        "launch": ("--session-new", "--trace", "--cuda-graph-trace", "--show-output"),
        "start": ("--session", "--output", "--sample", "--cpuctxsw"),
        "stop": ("--session",),
        "shutdown": ("--session", "--kill"),
        "stats": ("--report", "--format", "--force-export"),
    }.items():
        help_text, _ = control(nsys, [verb, "--help"], directory / f"nsys-{verb}-help.txt", env, timeout=15)
        supported = cli_options(help_text)
        missing = [option for option in options if option not in supported]
        if missing:
            raise RuntimeError(f"{version.strip()}: {verb} 缺少 {missing}；未启动 profiler")
    return version.strip()


def kernel_rows(text):
    rows = list(csv.reader(io.StringIO(text)))
    for i, row in enumerate(rows):
        if "Name" in row and any("Total Time" in field for field in row):
            names = row
            result = []
            for values in rows[i + 1:]:
                if len(values) == len(names):
                    record = dict(zip(names, values))
                    total_key = next(k for k in names if "Total Time" in k)
                    try:
                        if float(record[total_key].replace(",", "")) > 0:
                            result.append(record)
                    except ValueError:
                        pass
            return result
    return []


def profile(h, command, env, opener, infer, directory, *, mode=0):
    """infer() sends identical input and returns a validated response record.

    All signals target the pidfd of the wrapper we launched in a private nsys
    session. The wrapper execs llama-server without changing its PID.
    """
    directory.mkdir()
    nsys = find_nsys()
    if not nsys:
        return {"status": "unavailable", "reason": "未找到 nsys；GPU 性能采集未完成"}
    env = dict(env)
    env.update(GGML_CUDA_FLASHRT=str(mode), GGML_CUDA_FLASHRT_GGML_MATH="1",
               GGML_CUDA_FLASHRT_RMSNORM=str(mode), GGML_CUDA_FLASHRT_TRACE="0",
               GGML_CUDA_FLASHRT_SHADOW="0")
    env.pop("GGML_CUDA_DISABLE_GRAPHS", None)
    version = check_cli(nsys, directory, env)
    if h.listener_pid() is not None:
        raise RuntimeError("采集前 18081 被占用；未接管其他进程")
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise RuntimeError("采集需要 /usr/bin/python3 的 pidfd 支持")
    session = "flashrt" + uuid.uuid4().hex
    pidfile = directory / "server-pid.json"
    report = {"status": "running", "nsys": nsys, "version": version,
              "session": session, "mode": mode, "records": []}
    launcher = None
    fd = None
    started = False
    launch_log = directory / "server.log"

    def check_launcher():
        code = launcher.poll()
        report["launcher_returncode"] = code
        if code not in (None, 0):
            raise RuntimeError(f"nsys launch 退出 ({code})：" + launch_log.read_text(errors="replace")[-3000:])

    try:
        launch_args = [nsys, "launch", "--session-new=" + session,
                       "--trace=cuda,nvtx,cublas,osrt", "--cuda-graph-trace=node",
                       "--show-output=true",
                       sys.executable, str(Path(__file__).resolve()), "--server-exec", str(pidfile), *command]
        report["launch_command"] = launch_args
        with launch_log.open("w") as log:
            launcher = subprocess.Popen(launch_args, cwd=h.ROOT, env=env, stdin=subprocess.DEVNULL,
                                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + 120
            next_notice = 0
            while time.monotonic() < deadline:
                check_launcher()
                if fd is None and pidfile.exists():
                    pid, fd = owned_pidfd(pidfile)
                    report["server_pid"] = pid
                if fd is not None and select.select([fd], [], [], 0)[0]:
                    raise RuntimeError("采集服务提前退出")
                if fd is not None and h.listener_pid() == pid:
                    h.verify_server_identity(pid, Path(command[0]).resolve(), h.MODEL_DIR / "pi_llm.gguf",
                                             h.MODEL_DIR / "mmproj-model-f16.gguf")
                    try:
                        ready = h.request(opener, "GET", "/health", timeout=2).get("status") == "ok"
                    except (OSError, ValueError):
                        ready = False
                    if ready:
                        break
                if time.monotonic() >= next_notice:
                    print("等待 profile 服务加载…", flush=True)
                    next_notice = time.monotonic() + 10
                time.sleep(.25)
            else:
                raise RuntimeError("profile 服务 120 秒内未就绪")

            def checked_infer():
                check_launcher()
                if h.listener_pid() != pid or select.select([fd], [], [], 0)[0]:
                    raise RuntimeError("采集过程中服务发生变化")
                record = infer()
                check_launcher()
                return record

            for i in range(5):
                checked_infer()
                print(f"profile warmup {i+1}/5（未采集）", flush=True)
            start_args = ["start", "--session=" + session, "--sample=none", "--cpuctxsw=none",
                          "--output=" + str(directory / "steady")]
            report["start_command"] = [nsys, *start_args]
            control(nsys, start_args,
                    directory / "start.log", env)
            started = True
            for i in range(10):
                report["records"].append(checked_infer())
                print(f"profile request {i+1}/10", flush=True)
            control(nsys, ["stop", "--session=" + session], directory / "stop.log", env)
            started = False
            check_launcher()
            rep = directory / "steady.nsys-rep"
            if not rep.is_file() or rep.stat().st_size == 0:
                raise RuntimeError("未生成 steady.nsys-rep")
            for name in ("cuda_gpu_kern_sum", "cuda_api_sum", "cuda_gpu_mem_time_sum"):
                stats, _ = control(nsys, ["stats", "--force-export=true", "--report=" + name, "--format=csv", str(rep)],
                                   directory / (name + ".csv"), env)
                check_launcher()
                if name == "cuda_gpu_kern_sum":
                    kernels = kernel_rows(stats)
                    if not kernels:
                        raise RuntimeError("报告没有 CUDA kernel 数据；不能把空采集记作成功")
                    report["kernels"] = kernels
                    print("\nGPU kernel 汇总前 20 行（采集耗时不作正常性能结果）：", flush=True)
                    for record in kernels[:20]:
                        print(json.dumps(record, ensure_ascii=False), flush=True)
            check_launcher()
            report.update(status="captured", trace=str(rep))
            return report
    except BaseException as exc:
        report.update(status="incomplete", error=str(exc))
        raise
    finally:
        cleanup = []
        # These control commands refer only to our UUID-named session.
        if launcher is not None:
            if started:
                try:
                    _, code = control(nsys, ["stop", "--session=" + session], directory / "cleanup-stop.log", env,
                                      timeout=60, required=False)
                    if code:
                        cleanup.append(f"本次 session 的 nsys stop 返回 {code}")
                except Exception as exc:
                    cleanup.append(str(exc))
            try:
                _, code = control(nsys, ["shutdown", "--session=" + session, "--kill=none"],
                                  directory / "shutdown.log", env, timeout=30, required=False)
                if code:
                    cleanup.append(f"本次 session 的 nsys shutdown 返回 {code}")
            except Exception as exc:
                cleanup.append(str(exc))
            # A launcher can fail after spawning the wrapper but before the
            # readiness loop opens its pidfd. Still clean up that owned child.
            if fd is None and pidfile.exists():
                try:
                    pid, fd = owned_pidfd(pidfile)
                except (ProcessLookupError, FileNotFoundError):
                    pass
                except Exception as exc:
                    cleanup.append(str(exc))
            if fd is not None:
                try:
                    if not select.select([fd], [], [], 0)[0]:
                        signal.pidfd_send_signal(fd, signal.SIGTERM)
                        if not select.select([fd], [], [], 15)[0]:
                            cleanup.append("本次 profile 服务在 SIGTERM 后未退出")
                except ProcessLookupError:
                    pass
                except Exception as exc:
                    cleanup.append(str(exc))
                finally:
                    os.close(fd)
            try:
                code = launcher.wait(timeout=15)
                report["launcher_returncode"] = code
                if code:
                    cleanup.append(f"本次 nsys launch 返回 {code}")
            except subprocess.TimeoutExpired:
                cleanup.append("本次 nsys launch 仍未退出")
        if cleanup:
            report.update(status="incomplete", cleanup_errors=cleanup)
        h.write_json(directory / "summary.json", report)


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--server-exec":
        server_exec(sys.argv[2], sys.argv[3:])
    else:
        raise SystemExit("请通过 run-sm89-sprint.py 调用；此文件也由 nsys 启动服务时使用。")
