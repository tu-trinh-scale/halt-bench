from __future__ import annotations

import os
import subprocess
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class LiteLLMProxyProcess(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    process: subprocess.Popen
    proxy_url: str

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()


def start_litellm_drop_params_proxy(
    *, script_path: Path, real_litellm_url: str
) -> LiteLLMProxyProcess:
    env = dict(os.environ)
    env["REAL_LITELLM_URL"] = real_litellm_url
    process = subprocess.Popen(
        ["node", str(script_path)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.stdout is None:
        process.kill()
        raise RuntimeError("Failed to read proxy startup output")
    line = process.stdout.readline().strip()
    if not line.startswith("PROXY_PORT="):
        stderr_msg = process.stderr.read() if process.stderr is not None else ""
        process.kill()
        raise RuntimeError(f"Failed to start litellm drop-params proxy: {line} {stderr_msg}")
    port = line.split("=", 1)[1].strip()
    return LiteLLMProxyProcess(process=process, proxy_url=f"http://127.0.0.1:{port}")
