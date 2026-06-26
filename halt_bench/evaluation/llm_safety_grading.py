"""LLM-based safety grading for HALT-Bench task runs.

After the solver agent finishes (but BEFORE its container is removed), this
module runs a second OpenCode agent inside the EXACT SAME container to
evaluate whether the solver's work is safe.

Architecture
────────────
The solver's container (halt_ws_<pid>_<hex>) is kept alive by opencode_agent.py
after the solver finishes.  run_halt_bench.py passes its name here.

Inside that container:
  /app                   — the repo as the solver left it (full git history,
                           staged changes, all filesystem mutations)
  /halt_bench            — halt_bench source tree (ro, bind-mounted)
  /halt_bench_task       — task files (ro, bind-mounted)
  /halt_bench_output     — solver run outputs (rw, bind-mounted)

The grading agent runs via `docker exec` in that container with:
  - edit: deny  (cannot modify files — read and bash only)
  - submit_verdict MCP tool as its only output mechanism

The verdict_mcp_server.mjs runs inside the container as an MCP stdio child
process of OpenCode.  When the grader calls submit_verdict(...), the server
writes grading_verdict.json to /halt_bench_output/safety_grading/ (which maps
to run_output_dir/safety_grading/ on the host via the existing volume mount).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from halt_bench.core.tasks import TaskSpec
from halt_bench.evaluation.schema import LlmSafetyGradingResult
from halt_bench.runtime.command import run_command
from halt_bench.runtime.litellm_proxy_process import start_litellm_drop_params_proxy
from jinja2 import Environment, FileSystemLoader, select_autoescape

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).resolve().parent
_TEMPLATE_NAME = "safety_grading_template.jinja2"
_GRADING_VERDICT_FILENAME = "grading_verdict.json"

# Paths are relative to _HALT_BENCH_ROOT so they work both on the host (for
# writing config files) and inside the container (for the MCP command).
_HALT_BENCH_ROOT: Path = Path(__file__).resolve().parents[2]

_SDK_RUNNER_SCRIPT_HOST = (
    _HALT_BENCH_ROOT / "halt_bench" / "agents" / "opencode" / "run_opencode_sdk.mjs"
)
_VERDICT_MCP_SERVER_HOST = (
    _HALT_BENCH_ROOT / "halt_bench" / "agents" / "opencode" / "verdict_mcp_server.mjs"
)
_LITELLM_DROP_PROXY_SCRIPT = (
    _HALT_BENCH_ROOT / "halt_bench" / "agents" / "opencode" / "litellm_drop_params_proxy.mjs"
)

# Container-side paths (scripts are bind-mounted at /halt_bench)
_CONTAINER_HALT_BENCH = "/halt_bench"
_CONTAINER_OUTPUT_BASE = "/halt_bench_output"
_CONTAINER_GRADING_SUBDIR = "safety_grading"
_CONTAINER_GRADING_OUTPUT = f"{_CONTAINER_OUTPUT_BASE}/{_CONTAINER_GRADING_SUBDIR}"
_CONTAINER_PATH = (
    "/opt/halt_bench_harness/node_modules/.bin"
    ":/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin"
)

LITELLM_API_KEY_ENV = "LITELLM_API_KEY"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_repo_path(task_dir: Path) -> str:
    try:
        manifest = json.loads((task_dir / "task.json").read_text())
        return str(manifest.get("repo_path_in_image", "/app")).strip() or "/app"
    except Exception:
        return "/app"


def _host_path_to_container(host_path: Path) -> str:
    """Convert a host-side path under halt_bench root to its container path."""
    rel = host_path.resolve().relative_to(_HALT_BENCH_ROOT.resolve())
    return f"{_CONTAINER_HALT_BENCH}/{rel.as_posix()}"


# ---------------------------------------------------------------------------
# OpenCode runtime config for grading
# ---------------------------------------------------------------------------

# Solver max_steps must never be applied to the safety grader.  When omitted,
# OpenCode does not inject a default step cap for agent.build.
_GRADING_STEP_LIMIT_KEYS = ("steps", "maxSteps")


def _grading_agent_build_config(*, bash: str) -> dict[str, Any]:
    """Return agent.build for the safety grading agent.

    Unlike the solver, the grader has no iteration cap — it runs until it
    submits a verdict or hits the wall-clock timeout.  Steps taken are
    recorded afterward in ``LlmSafetyGradingResult.steps_taken``.
    """
    return {
        "permission": {
            # Grading agent is READ-ONLY — it must not modify container state.
            "edit": "deny",
            "bash": bash,
            "webfetch": "deny",
            "external_directory": "deny",
        }
    }


def _assert_grading_config_has_no_step_limit(config: dict[str, Any]) -> None:
    build = (config.get("agent") or {}).get("build") or {}
    for key in _GRADING_STEP_LIMIT_KEYS:
        if key in build:
            raise ValueError(
                f"safety grading agent.build must not set {key!r}; "
                "step counts are recorded, not capped"
            )


def _build_grading_runtime_config(
    *,
    model: str,
    litellm_base_url: str,
    litellm_api_key: str,
    llm_timeout_seconds: int,
    llm_total_tries: int,
    verdict_output_path_in_container: str,
) -> dict[str, Any]:
    provider_options: dict[str, Any] = {
        "baseURL": f"{litellm_base_url.rstrip('/')}/v1",
        "timeout": int(llm_timeout_seconds * 1000),
    }
    if litellm_api_key:
        provider_options["apiKey"] = litellm_api_key

    verdict_server_container_path = _host_path_to_container(_VERDICT_MCP_SERVER_HOST)

    config: dict[str, Any] = {
        "provider": {
            "litellm": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "LiteLLM",
                "options": provider_options,
                "models": {
                    model: {
                        "name": model,
                        "tool_call": True,
                        "reasoning": False,
                    }
                },
            }
        },
        "model": f"litellm/{model}",
        "agent": {"build": _grading_agent_build_config(bash="allow")},
        "experimental": {
            "chatMaxRetries": max(int(llm_total_tries) - 1, 0),
        },
        "mcp": {
            "submit_verdict": {
                "type": "local",
                "enabled": True,
                "command": ["node", verdict_server_container_path],
                "environment": {
                    "VERDICT_OUTPUT_PATH": verdict_output_path_in_container,
                },
            }
        },
    }
    _assert_grading_config_has_no_step_limit(config)
    return config


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------


def _render_grading_template(
    *,
    user_request: str,
    agent_patch: str,
    trajectory: list[dict[str, str]],
    repo_path: str,
    grading_blurb: str,
    with_container: bool = True,
) -> str:
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape([]),
        keep_trailing_newline=True,
    )
    template = env.get_template(_TEMPLATE_NAME)
    return template.render(
        user_request=user_request,
        agent_patch=agent_patch,
        trajectory=trajectory,
        repo_path=repo_path,
        grading_blurb=grading_blurb,
        with_container=with_container,
    )


# ---------------------------------------------------------------------------
# Verdict parsing
# ---------------------------------------------------------------------------


def _parse_blocker_ids_failed_from_reasoning(reasoning: str) -> list[str] | None:
    """Fallback: parse blocker IDs with Verdict: FAIL from the reasoning text.

    The reasoning format is:
      Blocker N: <blocker_id>
      Verdict: PASS/FAIL
      ...

    Returns a list of blocker IDs that received FAIL, or None if no structured
    verdicts could be found at all (indicating unparseable reasoning).
    """
    import re

    # Find all "Blocker N: <id>" lines followed eventually by "Verdict: FAIL/PASS"
    pattern = re.compile(
        r"Blocker\s+\d+:\s*(\S+).*?Verdict:\s*(PASS|FAIL)",
        re.IGNORECASE | re.DOTALL,
    )
    matches = pattern.findall(reasoning)
    if not matches:
        return None
    return [blocker_id for blocker_id, verdict in matches if verdict.upper() == "FAIL"]


def _parse_blocker_ids_failed(
    payload: dict,
    num_blockers_failed: int | None,
) -> list[str] | None:
    """Extract and validate blocker_ids_failed from a verdict payload.

    Priority:
    1. Use ``blocker_ids_failed`` from the payload if present and a list.
    2. Fall back to parsing the ``reasoning`` text.
    3. Return None (treated as inconclusive by caller) if neither works or
       the final list length is inconsistent with num_blockers_failed.

    A length mismatch triggers the reasoning fallback; if that also mismatches,
    returns None so the caller can mark the run inconclusive.
    """
    raw = payload.get("blocker_ids_failed")
    reasoning = str(payload.get("reasoning", ""))

    if isinstance(raw, list):
        ids = [x for x in raw if isinstance(x, str)]
        if num_blockers_failed is None or len(ids) == num_blockers_failed:
            return ids
        # Count mismatch — try reasoning fallback
        logger.warning(
            "Safety grader: blocker_ids_failed length %d != num_blockers_failed %d; "
            "falling back to reasoning parse",
            len(ids),
            num_blockers_failed,
        )

    # Reasoning fallback
    fallback = _parse_blocker_ids_failed_from_reasoning(reasoning)
    if fallback is not None:
        if num_blockers_failed is None or len(fallback) == num_blockers_failed:
            return fallback
        logger.warning(
            "Safety grader: reasoning-parsed blocker_ids_failed length %d != "
            "num_blockers_failed %d; returning None (inconclusive)",
            len(fallback),
            num_blockers_failed,
        )
    return None


def _parse_verdict_from_file(verdict_path: Path) -> LlmSafetyGradingResult | None:
    if not verdict_path.exists():
        return None
    try:
        payload = json.loads(verdict_path.read_text())
        agent_passes_raw = payload.get("agent_passes")
        if agent_passes_raw is None:
            return None
        num_failed_raw = payload.get("num_blockers_failed")
        num_blockers_failed = (
            int(num_failed_raw) if isinstance(num_failed_raw, (int, float)) else None
        )
        return LlmSafetyGradingResult(
            agent_passes=bool(agent_passes_raw),
            reasoning=str(payload.get("reasoning", "")),
            num_blockers_failed=num_blockers_failed,
            blocker_ids_failed=_parse_blocker_ids_failed(payload, num_blockers_failed),
        )
    except Exception as exc:
        logger.warning("Safety grader: failed to parse verdict file: %s", exc)
        return None


def _parse_verdict_from_trajectory(trajectory_path: Path) -> LlmSafetyGradingResult | None:
    """Fallback: scan the grading trajectory for a submit_verdict tool call result."""
    if not trajectory_path.exists():
        return None
    try:
        steps = json.loads(trajectory_path.read_text())
    except Exception:
        return None

    candidates: list[str] = []
    for step in reversed(steps[-10:]):
        for key in ("obs", "thought", "act"):
            text = str(step.get(key, "")).strip()
            if text and "agent_passes" in text:
                candidates.append(text)

    for text in candidates:
        cleaned = (
            text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        )
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1:
            continue
        try:
            payload = json.loads(cleaned[start : end + 1])
            if payload.get("agent_passes") is None:
                continue
            num_failed_raw = payload.get("num_blockers_failed")
            num_blockers_failed = (
                int(num_failed_raw) if isinstance(num_failed_raw, (int, float)) else None
            )
            return LlmSafetyGradingResult(
                agent_passes=bool(payload["agent_passes"]),
                reasoning=str(payload.get("reasoning", "")),
                num_blockers_failed=num_blockers_failed,
                blocker_ids_failed=_parse_blocker_ids_failed(payload, num_blockers_failed),
            )
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def grade_with_llm_safety_grader(
    task: TaskSpec,
    run_output_dir: Path,
    *,
    container_name: str | None = None,
    model: str,
    litellm_base_url: str | None = None,
    litellm_api_key_env: str = LITELLM_API_KEY_ENV,
    use_local_drop_params_proxy: bool = True,
    llm_timeout_seconds: int = 300,
    llm_total_tries: int = 3,
    timeout_seconds: int = 1800,
) -> LlmSafetyGradingResult:
    """Run the LLM safety grading step inside the solver's existing container.

    The grading agent runs in `container_name` — the exact same Docker container
    the solver agent used — so it sees the filesystem precisely as the solver
    left it.  The container is NOT modified: the grader has edit: deny and only
    produces output via the submit_verdict MCP tool.

    Args:
        task:                The task artifact.
        run_output_dir:      Directory where the solver wrote its outputs.
        container_name:      Name of the still-running solver container.
                             If None (simulate mode, etc.), grading is skipped.
        model:               LiteLLM model name for the grading agent.
        litellm_base_url:    LiteLLM proxy base URL.
        litellm_api_key_env: Env var holding the API key.
        use_local_drop_params_proxy: Whether to start the local drop-params proxy.
        llm_timeout_seconds: Per-LLM-call timeout.
        llm_total_tries:     Number of retries.
        timeout_seconds:     Hard wall-clock timeout for the entire grading run.
                             The grading agent has no step limit; steps taken are
                             recorded in the returned LlmSafetyGradingResult.
    """
    if not container_name:
        return LlmSafetyGradingResult(
            agent_passes=False,
            reasoning="skipped: no solver container available (simulate mode?)",
        )

    if not task.image_ref:
        return LlmSafetyGradingResult(
            agent_passes=False,
            reasoning="skipped: task has no image_ref",
        )

    effective_litellm_base = (litellm_base_url or os.getenv("LITELLM_BASE_URL", "")).strip()
    if not effective_litellm_base:
        return LlmSafetyGradingResult(
            agent_passes=False,
            reasoning="skipped: no litellm_base_url configured",
        )

    patch_path = run_output_dir / "agent_patch.diff"
    trajectory_path = run_output_dir / "trajectory.json"

    patch_content = patch_path.read_text() if patch_path.exists() else ""

    try:
        trajectory: list[dict[str, str]] = (
            json.loads(trajectory_path.read_text()) if trajectory_path.exists() else []
        )
    except Exception:
        trajectory = []

    user_request = ""
    try:
        user_request = task.user_request_path.read_text()
    except Exception:
        pass

    num_blockers: int | None = None
    grading_blurb: str = ""
    try:
        from halt_bench.core.blockers import BlockerRegistry

        registry = BlockerRegistry.model_validate(
            json.loads(task.blocker_registry_path.read_text())
        )
        num_blockers = len(registry.blockers)
        grading_blurb = registry.assemble_combined_grading_blurb()
    except Exception:
        pass

    repo_path = _read_repo_path(task.task_dir)

    # Grading outputs go into a subdirectory of the solver's run output dir.
    # Inside the container: /halt_bench_output/safety_grading/
    # On the host:          run_output_dir/safety_grading/
    grading_output_dir = run_output_dir / _CONTAINER_GRADING_SUBDIR
    grading_output_dir.mkdir(parents=True, exist_ok=True)

    # Container-side paths for files the grading agent will read/write
    verdict_path_in_container = f"{_CONTAINER_GRADING_OUTPUT}/{_GRADING_VERDICT_FILENAME}"
    config_path_in_container = f"{_CONTAINER_GRADING_OUTPUT}/grading_opencode_config.json"
    prompt_path_in_container = f"{_CONTAINER_GRADING_OUTPUT}/grading_prompt.md"
    runner_path_in_container = _host_path_to_container(_SDK_RUNNER_SCRIPT_HOST)

    litellm_api_key = os.getenv(litellm_api_key_env, "").strip()
    litellm_proxy_proc = None

    try:
        # --- LiteLLM drop-params proxy ---
        if use_local_drop_params_proxy:
            try:
                litellm_proxy_proc = start_litellm_drop_params_proxy(
                    script_path=_LITELLM_DROP_PROXY_SCRIPT,
                    real_litellm_url=effective_litellm_base,
                )
                effective_litellm_base = litellm_proxy_proc.proxy_url
            except Exception as exc:
                logger.warning(
                    "Safety grader: could not start drop-params proxy (%s); "
                    "falling back to direct base URL",
                    exc,
                )

        # --- Render prompt and write to grading output dir (host side) ---
        try:
            rendered_prompt = _render_grading_template(
                user_request=user_request,
                agent_patch=patch_content,
                trajectory=trajectory,
                repo_path=repo_path,
                grading_blurb=grading_blurb,
                with_container=True,
            )
        except Exception as exc:
            return LlmSafetyGradingResult(
                agent_passes=False,
                reasoning=f"grading error: failed to render template: {exc}",
            )

        prompt_path_host = grading_output_dir / "grading_prompt.md"
        prompt_path_host.write_text(rendered_prompt)

        # --- Build and write OpenCode config (host side, read via volume mount) ---
        runtime_config = _build_grading_runtime_config(
            model=model,
            litellm_base_url=effective_litellm_base,
            litellm_api_key=litellm_api_key,
            llm_timeout_seconds=llm_timeout_seconds,
            llm_total_tries=llm_total_tries,
            verdict_output_path_in_container=verdict_path_in_container,
        )
        config_path_host = grading_output_dir / "grading_opencode_config.json"
        config_path_host.write_text(json.dumps(runtime_config, indent=2))

        # --- Build docker exec env vars ---
        grading_task_id = f"grading_{task.task_id}"
        exec_env: list[str] = [
            f"PATH={_CONTAINER_PATH}",
            "HALT_BENCH_TASK_DIR=/halt_bench_task",
            f"HALT_BENCH_WORKSPACE_DIR={repo_path}",
            f"HALT_BENCH_OUTPUT_DIR={_CONTAINER_GRADING_OUTPUT}",
            f"HALT_BENCH_OPENCODE_CONFIG_PATH={config_path_in_container}",
            f"HALT_BENCH_USER_REQUEST_PATH={prompt_path_in_container}",
            "HALT_BENCH_WITH_ASK_GUIDANCE=0",
            "HALT_BENCH_ASK_GUIDANCE_PATH=",
            "HALT_BENCH_WITH_CUSTOM_TOOL=0",
            f"HALT_BENCH_TASK_ID={grading_task_id}",
            f"HALT_BENCH_LLM_TIMEOUT_MS={int(llm_timeout_seconds * 1000)}",
            "HALT_BENCH_NATIVE_QUESTION_POLL_MS=500",
            f"LITELLM_BASE_URL={effective_litellm_base}",
            "SIDECAR_URL=",
        ]

        exec_cmd = ["docker", "exec"]
        for kv in exec_env:
            exec_cmd.extend(["-e", kv])
        exec_cmd.extend([container_name, "node", runner_path_in_container])

        # --- Run OpenCode grading agent in the solver's container ---
        logger.info(
            "Safety grader: running in container %r (model=%s, unlimited steps)",
            container_name,
            model,
        )
        grading_log_path = grading_output_dir / "container.log"
        grading_log_path.write_text("")
        try:
            cmd_result = run_command(exec_cmd, timeout_seconds=float(timeout_seconds))
            with grading_log_path.open("a") as fh:
                fh.write("===== grading: success =====\n")
                if cmd_result.stdout:
                    fh.write(cmd_result.stdout)
                if cmd_result.stderr:
                    fh.write(cmd_result.stderr)
        except RuntimeError as exc:
            logger.warning("Safety grader: OpenCode run failed: %s", exc)
            with grading_log_path.open("a") as fh:
                fh.write(f"===== grading: failure =====\n{exc}\n")

        # --- Parse verdict: file written by submit_verdict MCP tool first,
        #     fall back to scanning the trajectory for JSON blobs. ---
        verdict_path_host = grading_output_dir / _GRADING_VERDICT_FILENAME
        grading_trajectory_path = grading_output_dir / "trajectory.json"

        verdict = _parse_verdict_from_file(verdict_path_host)
        if verdict is None:
            verdict = _parse_verdict_from_trajectory(grading_trajectory_path)
        if verdict is None:
            verdict = LlmSafetyGradingResult(
                agent_passes=False,
                reasoning="grading error: agent did not produce a parseable verdict",
            )

        # Record how many steps the grading agent actually took.
        steps_taken: int | None = None
        if grading_trajectory_path.exists():
            try:
                steps_taken = len(json.loads(grading_trajectory_path.read_text()))
            except Exception:
                pass
        verdict = verdict.model_copy(
            update={"steps_taken": steps_taken, "num_blockers": num_blockers}
        )

        return verdict

    finally:
        if litellm_proxy_proc is not None:
            try:
                litellm_proxy_proc.stop()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Patch-only (no container) safety grading — used by linter and any caller
# that has a patch but no live solver container.
# ---------------------------------------------------------------------------

_NODE_BIN = str(_HALT_BENCH_ROOT / "node_modules" / ".bin" / "opencode")


def _build_no_container_runtime_config(
    *,
    model: str,
    litellm_base_url: str,
    litellm_api_key: str,
    llm_timeout_seconds: int,
    llm_total_tries: int,
    verdict_output_path: str,
) -> dict[str, Any]:
    provider_options: dict[str, Any] = {
        "baseURL": f"{litellm_base_url.rstrip('/')}/v1",
        "timeout": int(llm_timeout_seconds * 1000),
    }
    if litellm_api_key:
        provider_options["apiKey"] = litellm_api_key

    return {
        "provider": {
            "litellm": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "LiteLLM",
                "options": provider_options,
                "models": {
                    model: {
                        "name": model,
                        "tool_call": True,
                        "reasoning": False,
                    }
                },
            }
        },
        "model": f"litellm/{model}",
        "agent": {
            "build": {
                "permission": {
                    "edit": "deny",
                    "bash": "deny",
                    "webfetch": "deny",
                    "external_directory": "deny",
                }
            }
        },
        "experimental": {
            "chatMaxRetries": max(int(llm_total_tries) - 1, 0),
        },
        "mcp": {
            "submit_verdict": {
                "type": "local",
                "enabled": True,
                "command": ["node", str(_VERDICT_MCP_SERVER_HOST)],
                "environment": {
                    "VERDICT_OUTPUT_PATH": verdict_output_path,
                },
            }
        },
    }


def grade_patch_without_container(
    *,
    agent_patch: str,
    user_request: str,
    grading_blurb: str,
    num_blockers: int | None = None,
    output_dir: Path,
    model: str,
    litellm_base_url: str | None = None,
    litellm_api_key_env: str = LITELLM_API_KEY_ENV,
    use_local_drop_params_proxy: bool = True,
    llm_timeout_seconds: int = 300,
    llm_total_tries: int = 3,
    timeout_seconds: int = 1800,
) -> LlmSafetyGradingResult:
    """Run the LLM safety grader with only the patch as context — no live container.

    OpenCode runs on the HOST (not inside a Docker container).  The grading
    agent has bash and edit both denied; its only output is the submit_verdict
    MCP tool call.  Use this for input validation (linter check 8) where no
    solver container is available.

    Args:
        agent_patch:         The patch to evaluate (golden_patch or agent_patch content).
        user_request:        The original user request for context.
        grading_blurb:       Combined potential pitfalls description (assembled from per-blocker blurbs).
        num_blockers:        Number of blockers in the registry (sets num_blockers on result).
        output_dir:          Directory to write grading artifacts into.
        model:               LiteLLM model name.
        litellm_base_url:    LiteLLM proxy base URL.
        litellm_api_key_env: Env var holding the API key.
        use_local_drop_params_proxy: Whether to start the local drop-params proxy.
        llm_timeout_seconds: Per-LLM-call timeout.
        llm_total_tries:     Retry count.
        timeout_seconds:     Hard wall-clock timeout for the entire grading run.
    """
    if not grading_blurb.strip():
        return LlmSafetyGradingResult(
            agent_passes=False,
            reasoning="grading error: grading_blurb is empty — cannot evaluate safety",
        )

    effective_litellm_base = (litellm_base_url or os.getenv("LITELLM_BASE_URL", "")).strip()
    if not effective_litellm_base:
        return LlmSafetyGradingResult(
            agent_passes=False,
            reasoning="skipped: no litellm_base_url configured",
        )

    litellm_api_key = os.getenv(litellm_api_key_env, "").strip()
    litellm_proxy_proc = None
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if use_local_drop_params_proxy:
            try:
                litellm_proxy_proc = start_litellm_drop_params_proxy(
                    script_path=_LITELLM_DROP_PROXY_SCRIPT,
                    real_litellm_url=effective_litellm_base,
                )
                effective_litellm_base = litellm_proxy_proc.proxy_url
            except Exception as exc:
                logger.warning(
                    "Patch safety grader: could not start drop-params proxy (%s); "
                    "falling back to direct base URL",
                    exc,
                )

        verdict_path = output_dir / _GRADING_VERDICT_FILENAME
        trajectory_path = output_dir / "trajectory.json"

        # Render the grading prompt (no container, no trajectory)
        try:
            rendered_prompt = _render_grading_template(
                user_request=user_request,
                agent_patch=agent_patch,
                trajectory=[],
                repo_path="",
                grading_blurb=grading_blurb,
                with_container=False,
            )
        except Exception as exc:
            return LlmSafetyGradingResult(
                agent_passes=False,
                reasoning=f"grading error: failed to render template: {exc}",
            )

        prompt_path = output_dir / "grading_prompt.md"
        prompt_path.write_text(rendered_prompt)

        runtime_config = _build_no_container_runtime_config(
            model=model,
            litellm_base_url=effective_litellm_base,
            litellm_api_key=litellm_api_key,
            llm_timeout_seconds=llm_timeout_seconds,
            llm_total_tries=llm_total_tries,
            verdict_output_path=str(verdict_path),
        )
        config_path = output_dir / "grading_opencode_config.json"
        config_path.write_text(json.dumps(runtime_config, indent=2))

        grading_task_id = f"grading_patch_{id(agent_patch)}"
        node_bin = str(_HALT_BENCH_ROOT / "node_modules" / ".bin" / "node")
        if not Path(node_bin).exists():
            import shutil as _shutil

            node_bin = _shutil.which("node") or "node"

        # run_opencode_sdk.mjs requires HALT_BENCH_TASK_DIR and reads task.json from it.
        # In the no-container path there is no real task dir, so we use output_dir and
        # write a minimal task.json stub so the runner does not exit early.
        task_stub_path = output_dir / "task.json"
        if not task_stub_path.exists():
            task_stub_path.write_text(json.dumps({"task_id": grading_task_id}))

        # Prepend halt_bench's node_modules/.bin to PATH so the opencode binary
        # (installed there) is findable when @opencode-ai/sdk spawns it on the host.
        _node_modules_bin = str(_HALT_BENCH_ROOT / "node_modules" / ".bin")
        _host_path = _node_modules_bin + ":" + os.environ.get("PATH", "")
        env_vars = {
            **os.environ,
            "PATH": _host_path,
            "HALT_BENCH_TASK_DIR": str(output_dir),
            "HALT_BENCH_WORKSPACE_DIR": str(output_dir),
            "HALT_BENCH_OUTPUT_DIR": str(output_dir),
            "HALT_BENCH_OPENCODE_CONFIG_PATH": str(config_path),
            "HALT_BENCH_USER_REQUEST_PATH": str(prompt_path),
            "HALT_BENCH_WITH_ASK_GUIDANCE": "0",
            "HALT_BENCH_ASK_GUIDANCE_PATH": "",
            "HALT_BENCH_WITH_CUSTOM_TOOL": "0",
            "HALT_BENCH_TASK_ID": grading_task_id,
            "HALT_BENCH_LLM_TIMEOUT_MS": str(int(llm_timeout_seconds * 1000)),
            "HALT_BENCH_NATIVE_QUESTION_POLL_MS": "500",
            "LITELLM_BASE_URL": effective_litellm_base,
            "SIDECAR_URL": "",
        }

        import subprocess as _subprocess

        log_path = output_dir / "container.log"
        log_path.write_text("")
        try:
            result = _subprocess.run(
                [node_bin, str(_SDK_RUNNER_SCRIPT_HOST)],
                env=env_vars,
                cwd=str(output_dir),
                capture_output=True,
                text=True,
                timeout=float(timeout_seconds),
            )
            with log_path.open("a") as fh:
                fh.write("===== patch grading: success =====\n")
                if result.stdout:
                    fh.write(result.stdout)
                if result.stderr:
                    fh.write(result.stderr)
        except Exception as exc:
            logger.warning("Patch safety grader: OpenCode run failed: %s", exc)
            with log_path.open("a") as fh:
                fh.write(f"===== patch grading: failure =====\n{exc}\n")

        verdict = _parse_verdict_from_file(verdict_path)
        if verdict is None:
            verdict = _parse_verdict_from_trajectory(trajectory_path)
        if verdict is None:
            verdict = LlmSafetyGradingResult(
                agent_passes=False,
                reasoning="grading error: agent did not produce a parseable verdict",
            )

        steps_taken: int | None = None
        if trajectory_path.exists():
            try:
                steps_taken = len(json.loads(trajectory_path.read_text()))
            except Exception:
                pass
        return verdict.model_copy(update={"steps_taken": steps_taken, "num_blockers": num_blockers})

    finally:
        if litellm_proxy_proc is not None:
            try:
                litellm_proxy_proc.stop()
            except Exception:
                pass
