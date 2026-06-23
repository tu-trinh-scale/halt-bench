from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

# Maps model name prefix → (litellm provider prefix, API key env var).
# Checked in order; first match wins.
_PROVIDER_PREFIXES: list[tuple[str, str, str]] = [
    ("claude", "anthropic", "ANTHROPIC_API_KEY"),
    ("gpt", "openai", "OPENAI_API_KEY"),
    ("o1", "openai", "OPENAI_API_KEY"),
    ("o3", "openai", "OPENAI_API_KEY"),
    ("o4", "openai", "OPENAI_API_KEY"),
    ("gemini", "gemini", "GEMINI_API_KEY"),
    ("mistral", "mistral", "MISTRAL_API_KEY"),
]


def infer_provider_for_model(model: str) -> tuple[str, str]:
    """Return ``(litellm_provider_model, api_key_env_var)`` for *model*.

    E.g. ``"claude-opus-4-6"`` → ``("anthropic/claude-opus-4-6", "ANTHROPIC_API_KEY")``.
    Falls back to passing the model name through unchanged if no prefix matches.
    """
    lower = model.lower()
    for prefix, provider, key_env in _PROVIDER_PREFIXES:
        if lower.startswith(prefix):
            return f"{provider}/{model}", key_env
    return model, "PROVIDER_API_KEY"


def _resolve_port(port: int) -> int:
    if port > 0:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _write_litellm_config(
    model_alias: str,
    provider_model: str,
    api_key_env: str,
    config_path: Path,
) -> None:
    """Write a minimal LiteLLM proxy config YAML to *config_path*."""
    config = {
        "model_list": [
            {
                "model_name": model_alias,
                "litellm_params": {
                    "model": provider_model,
                    "api_key": f"os.environ/{api_key_env}",
                    "drop_params": True,
                },
            }
        ],
        "litellm_settings": {
            "drop_params": True,
        },
    }
    with config_path.open("w") as fh:
        yaml.dump(config, fh, default_flow_style=False)


def _wait_for_litellm(base_url: str, timeout_s: int = 45) -> None:
    import urllib.request
    from urllib.error import URLError

    deadline = time.time() + timeout_s
    health_url = f"{base_url}/health/liveliness"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=2) as resp:
                if resp.status < 500:
                    return
        except (URLError, OSError):
            pass
        time.sleep(0.5)
    raise TimeoutError(
        f"Local LiteLLM proxy did not become ready at {base_url} within {timeout_s}s"
    )


class LiteLLMDirectProcess(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    process: subprocess.Popen
    url: str
    config_path: Path

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
        try:
            self.config_path.unlink(missing_ok=True)
        except Exception:
            pass


def start_litellm_direct_proxy(
    *,
    model: str,
    port: int = 0,
) -> LiteLLMDirectProcess:
    """Start a local LiteLLM proxy that routes *model* to the appropriate provider.

    The proxy reads the provider API key from the corresponding env var (e.g.
    ``ANTHROPIC_API_KEY`` for Claude models).  Returns a :class:`LiteLLMDirectProcess`
    whose ``url`` can be used as ``LITELLM_BASE_URL`` and whose ``api_key`` should
    be used as ``LITELLM_API_KEY`` when connecting to this local proxy.

    Args:
        model: Short model name used by the agent (e.g. ``"claude-opus-4-6"``).
        port: Port to bind to; 0 picks a free port automatically.
    """
    provider_model, api_key_env = infer_provider_for_model(model)
    resolved_port = _resolve_port(port)

    config_fd, config_path_str = tempfile.mkstemp(prefix="haltbench_litellm_", suffix=".yaml")
    os.close(config_fd)
    config_path = Path(config_path_str)
    _write_litellm_config(model, provider_model, api_key_env, config_path)

    env = dict(os.environ)
    env["LITELLM_CONFIG_PATH"] = str(config_path)

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "litellm.proxy.proxy_server:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(resolved_port),
            "--log-level",
            "warning",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    url = f"http://127.0.0.1:{resolved_port}"
    try:
        _wait_for_litellm(url)
    except Exception as exc:
        process.terminate()
        stderr_tail = ""
        try:
            _, err = process.communicate(timeout=2)
            stderr_tail = (err or "")[-3000:]
        except Exception:
            pass
        config_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to start local LiteLLM proxy for model '{model}' "
            f"(provider model: '{provider_model}', api_key_env: '{api_key_env}'): {exc}"
            + (f"\nstderr tail:\n{stderr_tail}" if stderr_tail.strip() else "")
        ) from exc

    return LiteLLMDirectProcess(
        process=process,
        url=url,
        config_path=config_path,
    )
