from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Mapping

from pydantic import BaseModel


_REDACTED = "<REDACTED>"
_DOCKER_ENV_FLAG_NAMES = {"-e", "--env"}
_SENSITIVE_FLAG_NAMES = {
    "--api-key",
    "--api_key",
    "--access-token",
    "--access_token",
    "--auth-token",
    "--auth_token",
    "--bearer-token",
    "--bearer_token",
    "--client-secret",
    "--client_secret",
    "--password",
    "--secret",
    "--token",
}
_SENSITIVE_ENV_KEYS = {
    "API_KEY",
    "AUTH_TOKEN",
    "BEARER_TOKEN",
    "CLIENT_SECRET",
    "GITHUB_GIST_TOKEN",
    "GITHUB_TOKEN",
    "LITELLM_API_KEY",
    "OPENAI_API_KEY",
    "PASSWORD",
    "SECRET",
    "TOKEN",
}
_SENSITIVE_TEXT_KEY_PATTERN = (
    r"[A-Za-z0-9_]*(?:API[_-]?KEY|TOKEN|SECRET|PASSWORD)"
    r"|apiKey|api_key|auth[_-]?token|bearer[_-]?token|client[_-]?secret"
)


def _is_sensitive_env_key(key: str) -> bool:
    normalized = key.strip().upper()
    return (
        normalized in _SENSITIVE_ENV_KEYS
        or normalized.endswith("_API_KEY")
        or normalized.endswith("_TOKEN")
        or normalized.endswith("_SECRET")
        or normalized.endswith("_PASSWORD")
    )


def _redact_assignment(arg: str) -> str:
    key, sep, _value = arg.partition("=")
    if sep and _is_sensitive_env_key(key):
        return f"{key}={_REDACTED}"
    return arg


def _is_docker_login(args: list[str]) -> bool:
    if not args or str(args[0]) != "docker":
        return False
    return any(str(arg) == "login" for arg in args[1:4])


def _to_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def _redact_text(value) -> str:
    text = _to_text(value)
    if not text:
        return text

    # Raw LiteLLM-proxy token shape seen in artifacts.
    text = re.sub(r"LLM\|[^\s\"'`,;]+", f"LLM|{_REDACTED}", text)

    # Authorization headers and similar bearer-token snippets.
    text = re.sub(
        r"(?i)(\bBearer\s+)[A-Za-z0-9._~+/=-]+",
        rf"\1{_REDACTED}",
        text,
    )

    # JSON/YAML/shell-ish quoted assignments: "apiKey": "sk-...", TOKEN='...'
    quoted_pattern = re.compile(
        rf"(?i)([\"']?(?:{_SENSITIVE_TEXT_KEY_PATTERN})[\"']?\s*[:=]\s*)([\"'])([^\"']*)(\2)"
    )
    text = quoted_pattern.sub(lambda match: f"{match.group(1)}{match.group(2)}{_REDACTED}{match.group(4)}", text)

    # Unquoted assignments: LITELLM_API_KEY=sk-..., api_key: sk-...
    unquoted_pattern = re.compile(
        rf"(?i)([\"']?(?:{_SENSITIVE_TEXT_KEY_PATTERN})[\"']?\s*[:=]\s*)([^\s,}}]+)"
    )
    text = unquoted_pattern.sub(lambda match: f"{match.group(1)}{_REDACTED}", text)

    return text


def _redact_args(args: list[str]) -> str:
    redacted: list[str] = []
    redact_next = False
    redact_env_next = False
    docker_login = _is_docker_login(args)
    for raw_arg in args:
        arg = str(raw_arg)
        if redact_next:
            redacted.append(_REDACTED)
            redact_next = False
            continue
        if redact_env_next:
            redacted.append(_redact_assignment(arg))
            redact_env_next = False
            continue

        flag_name, sep, _flag_value = arg.partition("=")
        normalized_flag = flag_name.strip().lower()
        if normalized_flag in _DOCKER_ENV_FLAG_NAMES:
            if sep:
                redacted.append(f"{flag_name}={_redact_assignment(_flag_value)}")
            else:
                redacted.append(arg)
                redact_env_next = True
            continue
        if arg.startswith("-e") and arg != "-e":
            redacted.append("-e" + _redact_assignment(arg[2:]))
            continue
        if docker_login and (normalized_flag in {"-p", "--password"} or arg.startswith("-p")):
            if sep:
                redacted.append(f"{flag_name}={_REDACTED}")
            elif arg.startswith("-p") and arg != "-p":
                redacted.append("-p" + _REDACTED)
            else:
                redacted.append(arg)
                redact_next = True
            continue
        if normalized_flag in _SENSITIVE_FLAG_NAMES:
            redacted.append(f"{flag_name}={_REDACTED}" if sep else arg)
            redact_next = not sep
            continue

        redacted.append(_redact_assignment(arg))

    return _redact_text(" ".join(redacted))


class CommandResult(BaseModel):
    stdout: str
    stderr: str
    returncode: int


def run_command(
    args: list[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    check: bool = True,
    timeout_seconds: float | None = None,
) -> CommandResult:
    try:
        completed = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            env=dict(env) if env else None,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Command timed out after {timeout_seconds}s: {_redact_args(args)}\n"
            f"stdout:\n{_redact_text(exc.stdout)}\n"
            f"stderr:\n{_redact_text(exc.stderr)}"
        ) from exc
    result = CommandResult(
        stdout=completed.stdout,
        stderr=completed.stderr,
        returncode=completed.returncode,
    )
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {_redact_args(args)}\n"
            f"stdout:\n{_redact_text(completed.stdout)}\n"
            f"stderr:\n{_redact_text(completed.stderr)}"
        )
    return result
