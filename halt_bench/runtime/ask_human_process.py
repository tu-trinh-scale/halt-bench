from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from pydantic import BaseModel, ConfigDict


def _terminate_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


class AskHumanProcess(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    process: subprocess.Popen
    url: str

    def stop(self) -> None:
        _terminate_process(self.process)


def start_ask_human_process(
    *,
    tasks_dir: Path,
    port: int,
    provider: str,
    max_retries: int = 3,
) -> AskHumanProcess:
    project_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH", "")
    if existing_pythonpath:
        env["PYTHONPATH"] = f"{project_root}:{existing_pythonpath}"
    else:
        env["PYTHONPATH"] = str(project_root)
    env["ASK_HUMAN_PROVIDER"] = provider
    # Skip pydantic's plugin discovery scan, which reads every installed package's
    # entry_points.txt from EFS and can hang indefinitely under I/O pressure.
    # The ask_human server uses no pydantic plugins, so this is safe.
    env.setdefault("PYDANTIC_DISABLE_PLUGINS", "1")

    last_exc: Exception | None = None
    for attempt_num in range(max_retries):
        # On retry always request a fresh dynamic port (ignore any caller-supplied
        # fixed port after the first attempt to avoid reusing the conflicted one).
        resolved_port = _resolve_port(port if attempt_num == 0 else 0)
        args = [
            sys.executable,
            "-m",
            "halt_bench.ask_human.server",
            "--tasks-dir",
            str(tasks_dir),
            "--port",
            str(resolved_port),
        ]
        process = subprocess.Popen(
            args,
            env=env,
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        url = f"http://127.0.0.1:{resolved_port}"
        # Quick startup check: if the subprocess exits within 1.5 s it almost
        # certainly crashed (e.g. "Address already in use").  Detect this early
        # so we don't sit through the full 45-s _wait_for_server timeout.
        time.sleep(1.5)
        if process.poll() is not None:
            try:
                _, err = process.communicate(timeout=1)
            except Exception:
                err = ""
            err_text = (err or "")[-2000:]
            if attempt_num < max_retries - 1 and (
                "Address already in use" in err_text
                or "OSError" in err_text
                or "Errno 98" in err_text
            ):
                # Port conflict — release and retry with a fresh port.
                _terminate_process(process)
                continue
            last_exc = TimeoutError(
                f"ask_human server did not become ready at {url}\nstderr_tail:\n{err_text.strip()}"
            )
            break

        try:
            _wait_for_server(url)
            return AskHumanProcess(process=process, url=url)
        except Exception as exc:
            _terminate_process(process)
            stdout_tail = ""
            stderr_tail = ""
            try:
                out, err = process.communicate(timeout=1)
                stdout_tail = (out or "")[-4000:]
                stderr_tail = (err or "")[-4000:]
            except Exception:
                pass
            details = []
            if stdout_tail.strip():
                details.append(f"stdout_tail:\n{stdout_tail.strip()}")
            if stderr_tail.strip():
                details.append(f"stderr_tail:\n{stderr_tail.strip()}")
            # If the process crashed (likely port conflict) and we have retries left,
            # loop back and try a new port.
            if (
                attempt_num < max_retries - 1
                and process.poll() is not None
                and any(
                    marker in "\n".join(details)
                    for marker in ("Address already in use", "OSError", "Errno 98")
                )
            ):
                last_exc = exc
                continue
            if details:
                raise TimeoutError(f"{exc}\n" + "\n".join(details)) from exc
            raise
    # All retries exhausted.
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("ask_human server failed to start after retries")


def _resolve_port(port: int) -> int:
    if port > 0 and _is_port_available(port):
        return port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _is_port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) != 0


def _wait_for_server(base_url: str, timeout_s: int = 120) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            req = urlopen(base_url + "/get_logs", data=b"{}", timeout=2)
            if req.status < 500:
                return
        except URLError:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"ask_human server did not become ready at {base_url}")
