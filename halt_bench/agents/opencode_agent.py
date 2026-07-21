from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import subprocess
import time
import uuid
from pathlib import Path

from halt_bench.agents.base import AgentRunResult, HaltAgent
from halt_bench.agents.opencode.types import OpenCodeConfig, TrajectoryStep
from halt_bench.core.tasks import TaskSpec
from halt_bench.runtime.command import run_command
from halt_bench.runtime.litellm_proxy_process import start_litellm_drop_params_proxy

logger = logging.getLogger(__name__)

# Root of the halt_bench project tree (three parents above this file:
#   halt_bench/agents/opencode_agent.py → halt_bench/agents → halt_bench → <project_root>)
_HALT_BENCH_ROOT: Path = Path(__file__).resolve().parents[2]

# PATH inside the task container.
# /opt/halt_bench_harness/node_modules/.bin  — opencode CLI, sdk binaries (baked into task image)
# /usr/local/bin                             — task repo's node/npm runtime
# /opt/halt_bench_harness/bin/node20         — harness Node used explicitly for OpenCode
# standard Linux paths
_CONTAINER_PATH = (
    "/opt/halt_bench_harness/node_modules/.bin"
    ":/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin"
)
_HARNESS_NODE = "/opt/halt_bench_harness/bin/node20"


def _container_state_summary(container_name: str) -> str:
    inspect = subprocess.run(
        [
            "docker",
            "inspect",
            container_name,
            "--format",
            "status={{.State.Status}} running={{.State.Running}} "
            "exit={{.State.ExitCode}} oom={{.State.OOMKilled}} error={{.State.Error}}",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if inspect.returncode != 0:
        return inspect.stderr.strip()

    logs = subprocess.run(
        ["docker", "logs", "--tail", "40", container_name],
        capture_output=True,
        text=True,
        timeout=15,
    )
    log_text = (logs.stdout + logs.stderr).strip()
    if log_text:
        return f"{inspect.stdout.strip()} logs={log_text}"
    return inspect.stdout.strip()


def _container_is_running(container_name: str) -> bool:
    inspect = subprocess.run(
        ["docker", "inspect", container_name, "--format", "{{.State.Running}}"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    return inspect.returncode == 0 and inspect.stdout.strip() == "true"


def _read_repo_path_in_image(task_dir: Path) -> str:
    """Read repo_path_in_image from task.json, defaulting to /app."""
    try:
        manifest = json.loads((task_dir / "task.json").read_text())
        return str(manifest.get("repo_path_in_image", "/app")).strip() or "/app"
    except Exception:
        return "/app"


def _host_path_to_container(
    host_path: Path,
    output_dir: Path,
    task_dir: Path,
) -> str | None:
    """Map a host filesystem path to its equivalent inside the task container.

    Active volume mounts:
      /halt_bench        → _HALT_BENCH_ROOT  (ro)  harness scripts
      /halt_bench_task   → task_dir          (ro)  task files
      /halt_bench_output → output_dir        (rw)  run outputs
    """
    try:
        rel = host_path.resolve().relative_to(_HALT_BENCH_ROOT.resolve())
        return f"/halt_bench/{rel}"
    except ValueError:
        pass
    try:
        rel = host_path.resolve().relative_to(output_dir.resolve())
        return f"/halt_bench_output/{rel}"
    except ValueError:
        pass
    try:
        rel = host_path.resolve().relative_to(task_dir.resolve())
        return f"/halt_bench_task/{rel}"
    except ValueError:
        pass
    return None


def _start_task_container(
    image_ref: str,
    container_name: str,
    task_dir: Path,
    output_dir: Path,
) -> None:
    """Start a long-running task container.

    Volume layout inside the container
    ────────────────────────────────────────────────────────────────────────
    /halt_bench        ← _HALT_BENCH_ROOT   (ro)  agent .mjs / .py scripts
    /halt_bench_task   ← task_dir           (ro)  task files
    /halt_bench_output ← output_dir         (rw)  outputs written back to host
    ────────────────────────────────────────────────────────────────────────

    The OpenCode harness Node.js runtime is baked into the task image at
    /opt/halt_bench_harness/bin/node20, while the task repo's own node/npm may
    remain on PATH for repo-specific test compatibility.

    --network host lets the container reach 127.0.0.1-bound host services
    (LiteLLM proxy, ask_human sidecar) without URL rewriting.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    create_cmd = [
        "docker",
        "create",
        "--name",
        container_name,
        "--entrypoint",
        "",
        "--network",
        "host",
        "-v",
        f"{_HALT_BENCH_ROOT}:/halt_bench:ro",
        "-v",
        f"{task_dir}:/halt_bench_task:ro",
        "-v",
        f"{output_dir}:/halt_bench_output:rw",
        image_ref,
        "tail",
        "-f",
        "/dev/null",
    ]
    logger.info("Starting task container %r from %r", container_name, image_ref)

    last_error = ""
    for attempt in range(1, 4):
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, timeout=30)
        try:
            create_result = subprocess.run(create_cmd, capture_output=True, timeout=300)
            if create_result.returncode != 0:
                last_error = create_result.stderr.decode(errors="replace").strip()
                logger.warning(
                    "docker create failed for %r on attempt %d/3: %s",
                    container_name,
                    attempt,
                    last_error,
                )
                continue

            start_result = subprocess.run(
                ["docker", "start", container_name],
                capture_output=True,
                timeout=300,
            )
            if start_result.returncode == 0:
                for _ in range(10):
                    if _container_is_running(container_name):
                        logger.info("Task container %r started", container_name)
                        return
                    time.sleep(0.5)
                last_error = _container_state_summary(container_name)
                logger.warning(
                    "docker start returned success for %r on attempt %d/3, "
                    "but the container is not running: %s",
                    container_name,
                    attempt,
                    last_error,
                )
                continue
            last_error = start_result.stderr.decode(errors="replace").strip()
            logger.warning(
                "docker start failed for %r on attempt %d/3: %s",
                container_name,
                attempt,
                last_error,
            )
        except subprocess.TimeoutExpired as exc:
            last_error = f"{exc.cmd!r} timed out after {exc.timeout} seconds"
            logger.warning(
                "Docker container startup timed out for %r on attempt %d/3: %s",
                container_name,
                attempt,
                last_error,
            )

    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, timeout=30)
    raise RuntimeError(
        f"Failed to start task container {container_name!r} from {image_ref!r}: {last_error}"
    )


def _verify_node_in_container(container_name: str) -> None:
    """Verify Node.js is available in the task container."""
    last_stderr = ""
    for attempt in range(1, 6):
        if not _container_is_running(container_name):
            last_stderr = _container_state_summary(container_name)
            logger.warning(
                "Task container %r is not running before node verification "
                "on attempt %d/5: %s",
                container_name,
                attempt,
                last_stderr,
            )
            restart = subprocess.run(
                ["docker", "start", container_name],
                capture_output=True,
                timeout=60,
            )
            if restart.returncode != 0:
                last_stderr = restart.stderr.decode(errors="replace").strip()
            time.sleep(1)
            continue

        result = subprocess.run(
            [
                "docker",
                "exec",
                "-e",
                f"PATH={_CONTAINER_PATH}",
                container_name,
                "node",
                "--version",
            ],
            capture_output=True,
            timeout=15,
        )
        if result.returncode == 0:
            logger.debug("Node.js in %r: %s", container_name, result.stdout.decode().strip())
            return

        last_stderr = result.stderr.decode(errors="replace").strip()
        if any(
            marker in last_stderr
            for marker in (
                "is not running",
                "unable to upgrade to tcp",
                "No such exec instance",
                "context canceled",
            )
        ):
            time.sleep(1)
            continue

        break

    raise RuntimeError(
        f"Node.js not found in task container {container_name!r}. "
        f"Was the task image built with Dockerfile.task_runtime? "
        f"stderr: {last_stderr}; container_state: {_container_state_summary(container_name)}"
    )


def _read_error_details(result_path: Path, debug_path: Path) -> str:
    details: list[str] = []
    if result_path.exists():
        try:
            payload = json.loads(result_path.read_text())
            error = payload.get("error")
            if error:
                details.append(f"result.error: {error}")
        except Exception:
            pass
    if debug_path.exists():
        try:
            payload = json.loads(debug_path.read_text())
            errors = payload.get("errors") or []
            if isinstance(errors, list) and errors:
                details.append("sdk_debug.errors: " + " | ".join(str(e) for e in errors[-3:]))
        except Exception:
            pass
    return "\n".join(details)


def _is_sensitive_config_key(key: str) -> bool:
    normalized = key.strip().lower().replace("-", "_")
    return normalized in {
        "api_key",
        "apikey",
        "auth_token",
        "bearer_token",
        "client_secret",
        "password",
        "secret",
        "token",
    } or normalized.endswith(
        ("_api_key", "_token", "_secret", "_password", "_header", "_headers", "_cookie")
    )


def _redact_sensitive_config(value):
    if isinstance(value, dict):
        return {
            key: (
                "<REDACTED>"
                if _is_sensitive_config_key(str(key))
                else _redact_sensitive_config(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_sensitive_config(item) for item in value]
    return value


def _rewrite_json_file_redacted(path: Path) -> None:
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text())
        path.write_text(json.dumps(_redact_sensitive_config(payload), indent=2))
    except Exception:
        logger.warning("Failed to redact sensitive values from %s", path, exc_info=True)


class OpenCodeAgent(HaltAgent):
    """OpenCode agent that runs the solver inside a Docker task container.

    Architecture
    ────────────
    Task creation produces a single image per task:
      haltbench:<tag>  — task repo at base commit + Node 20 + opencode npm deps
                         (produced by SWEBenchProTaskCreator using Dockerfile.task_runtime)

    Each solve run spins up a fresh container from that image:
      - Task repo lives at its native path inside the image (e.g. /app).
      - Agent scripts are bind-mounted read-only from the host EFS at /halt_bench.
        Code changes deploy instantly without image rebuilds.
      - Outputs are written directly to /halt_bench_output (rw mount) — no docker cp.
      - LiteLLM proxy and ask_human sidecar run on the host; --network host gives the
        container direct access to 127.0.0.1-bound services.
    """

    def __init__(self, config: OpenCodeConfig | None = None):
        self.config = config or OpenCodeConfig()

    def run(
        self,
        task: TaskSpec,
        *,
        output_dir: Path,
        ask_human_url: str,
        user_request_override_path: Path | None = None,
    ) -> AgentRunResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        trajectory_path = output_dir / "trajectory.json"
        patch_path = output_dir / "agent_patch.diff"
        result_path = output_dir / "result.json"
        runtime_config_path = output_dir / "opencode_runtime_config.json"
        container_log_path = output_dir / "container.log"
        container_log_path.write_text("")

        if self.config.simulate:
            container_log_path.write_text("[simulate] no container execution\n")
            trajectory_path.write_text(
                json.dumps(
                    [
                        TrajectoryStep(
                            thought="Need clarification about hidden requirement.",
                            act="ask_human [custom] what format should the token use?",
                            obs="Use PAT format X",
                        ).model_dump(),
                        TrajectoryStep(
                            thought="Now update implementation with clarified requirement.",
                            act="shell: apply fix in auth module",
                            obs="done",
                        ).model_dump(),
                    ],
                    indent=2,
                )
            )
            patch_path.write_text("")
            result_path.write_text(json.dumps({"status": "simulated_success"}, indent=2))
            return AgentRunResult(
                success=True,
                trajectory_path=trajectory_path,
                patch_path=patch_path,
                result_path=result_path,
                metadata={"mode": "simulate", "container_log_path": str(container_log_path)},
            )

        # ── Container setup ────────────────────────────────────────────────────
        repo_path = _read_repo_path_in_image(task.task_dir)
        image_ref = task.image_ref or ""
        container_name = f"halt_ws_{os.getpid()}_{uuid.uuid4().hex[:12]}"

        last_error: str | None = None
        litellm_proxy_proc = None
        container_started = False
        run_succeeded = False

        try:
            _start_task_container(
                image_ref,
                container_name,
                task.task_dir,
                output_dir,
            )
            container_started = True
            _verify_node_in_container(container_name)

            # ── Runtime git setup ─────────────────────────────────────────────────
            # Behaviour depends on task.support_setup_patch:
            #   False (permanent) → no-op; setup_script.sh already committed
            #     everything at image-build time.
            #   True  (temporary/legacy) → nuclear-reset + re-run setup_script.sh
            #     to fix uncommitted setup_patch.diff contamination.
            # See README § "Runtime Git Setup: Permanent vs Temporary Pipeline".
            self._run_repo_setup_in_container(container_name, repo_path, task)

            # ── LiteLLM drop-params proxy (runs on host) ──────────────────────
            litellm_base = self.config.litellm_base_url or os.getenv("LITELLM_BASE_URL")
            litellm_api_key = os.getenv(self.config.litellm_api_key_env, "").strip()
            effective_litellm_base = ""

            if self.config.use_local_litellm_drop_params_proxy and litellm_base:
                litellm_proxy_proc = start_litellm_drop_params_proxy(
                    script_path=self.config.litellm_drop_proxy_script,
                    real_litellm_url=litellm_base,
                )
                effective_litellm_base = litellm_proxy_proc.proxy_url
            elif litellm_base:
                effective_litellm_base = litellm_base
            else:
                raise RuntimeError(
                    "OpenCode SDK runner requires a LiteLLM/OpenAI-compatible base URL. "
                    "Set --opencode-litellm-base-url or LITELLM_BASE_URL."
                )
            if not litellm_api_key and not self.config.litellm_direct:
                raise RuntimeError(
                    f"OpenCode SDK runner requires {self.config.litellm_api_key_env} in environment."
                )

            container_litellm_base = effective_litellm_base
            container_sidecar_url = ask_human_url

            # ── Build OpenCode runtime config ──────────────────────────────────
            opencode_runtime_config = self._build_opencode_runtime_config(
                task,
                container_sidecar_url,
                litellm_base_url=container_litellm_base,
                litellm_api_key=litellm_api_key,
                container_mode=True,
            )
            runtime_config_path.write_text(json.dumps(opencode_runtime_config, indent=2))

            # ── Resolve user request path (container-internal) ─────────────────
            container_user_request_path = "/halt_bench_task/user_request.md"
            if user_request_override_path is not None:
                mapped = _host_path_to_container(
                    user_request_override_path, output_dir, task.task_dir
                )
                if mapped is None:
                    dest = output_dir / user_request_override_path.name
                    shutil.copy2(user_request_override_path, dest)
                    mapped = f"/halt_bench_output/{user_request_override_path.name}"
                container_user_request_path = mapped

            # ── Resolve ask guidance path (container-internal) ─────────────────
            container_ask_guidance_path = ""
            if self.config.with_ask_guidance and self.config.ask_guidance_path:
                mapped = _host_path_to_container(
                    self.config.ask_guidance_path, output_dir, task.task_dir
                )
                if mapped is None:
                    dest = output_dir / self.config.ask_guidance_path.name
                    shutil.copy2(self.config.ask_guidance_path, dest)
                    mapped = f"/halt_bench_output/{self.config.ask_guidance_path.name}"
                container_ask_guidance_path = mapped

            # ── docker exec environment ────────────────────────────────────────
            container_env: dict[str, str] = {
                "HALT_BENCH_TASK_DIR": "/halt_bench_task",
                "HALT_BENCH_WORKSPACE_DIR": repo_path,
                "HALT_BENCH_OUTPUT_DIR": "/halt_bench_output",
                "HALT_BENCH_OPENCODE_CONFIG_PATH": "/halt_bench_output/opencode_runtime_config.json",
                "HALT_BENCH_USER_REQUEST_PATH": container_user_request_path,
                "HALT_BENCH_ASK_GUIDANCE_PATH": container_ask_guidance_path,
                "HALT_BENCH_MODEL": self.config.model,
                "HALT_BENCH_TASK_ID": task.task_id,
                "HALT_BENCH_WITH_CUSTOM_TOOL": "1" if self.config.with_custom_tool else "0",
                "HALT_BENCH_WITH_ASK_GUIDANCE": "1" if self.config.with_ask_guidance else "0",
                "HALT_BENCH_LLM_TIMEOUT_MS": str(int(self.config.llm_timeout_seconds * 1000)),
                # Set the JS wrapper's run timeout 5 minutes shorter than the Python
                # orchestrator's docker-exec timeout so the wrapper always exits cleanly
                # (writing result.json / trajectory) before the orchestrator force-kills it.
                "HALT_BENCH_RUN_TIMEOUT_MS": str(
                    max(60_000, int((self.config.run_timeout_seconds - 300) * 1000))
                    if self.config.run_timeout_seconds is not None
                    else 6_900_000
                ),
                "HALT_BENCH_NATIVE_QUESTION_POLL_MS": str(
                    self.config.native_question_poll_interval_ms
                ),
                "SIDECAR_URL": container_sidecar_url,
                "LITELLM_BASE_URL": container_litellm_base,
                "LITELLM_API_KEY": litellm_api_key,
                "OPENAI_API_KEY": litellm_api_key,
                "OPENCODE_NO_UPDATE": "1",
                "HOME": "/tmp",
                "OPENCODE_EXPERIMENTAL_DISABLE_FILEWATCHER": "1",
                "OPENCODE_DISABLE_MODELS_FETCH": "1",
                "OPENCODE_DISABLE_LSP_DOWNLOAD": "1",
                "OPENCODE_FAST_BOOT": "1",
                "PATH": _CONTAINER_PATH,
                # Used only by test mocks to locate the real output dir on the host.
                "HALT_BENCH_HOST_OUTPUT_DIR": str(output_dir),
            }

            # Runner script lives in the EFS-mounted /halt_bench tree.
            container_runner = (
                "/halt_bench/"
                + self.config.sdk_runner_script.relative_to(_HALT_BENCH_ROOT).as_posix()
            )
            exec_cmd = ["docker", "exec"]
            for key, val in container_env.items():
                exec_cmd.extend(["-e", f"{key}={val}"])
            runner_arg = shlex.quote(container_runner)
            harness_node_script = (
                f"export PATH={shlex.quote(_CONTAINER_PATH)}; "
                f"if [ -x {_HARNESS_NODE} ]; then exec {_HARNESS_NODE} {runner_arg}; fi; "
                "if command -v apk >/dev/null 2>&1 "
                "&& [ -x /opt/haltbench-glibc/ld-linux-x86-64.so.2 ]; then "
                "exec /opt/haltbench-glibc/ld-linux-x86-64.so.2 "
                "--library-path /opt/haltbench-glibc "
                f"/opt/node20/bin/node {runner_arg}; "
                "fi; "
                f"exec /opt/node20/bin/node {runner_arg}"
            )
            exec_cmd.extend([container_name, "sh", "-c", harness_node_script])

            try:
                command_result = run_command(
                    exec_cmd,
                    timeout_seconds=(
                        float(self.config.run_timeout_seconds)
                        if self.config.run_timeout_seconds is not None
                        else None
                    ),
                )
                with container_log_path.open("a") as handle:
                    handle.write("===== run: success =====\n")
                    if command_result.stdout:
                        handle.write("----- stdout -----\n")
                        handle.write(command_result.stdout)
                        if not command_result.stdout.endswith("\n"):
                            handle.write("\n")
                    if command_result.stderr:
                        handle.write("----- stderr -----\n")
                        handle.write(command_result.stderr)
                        if not command_result.stderr.endswith("\n"):
                            handle.write("\n")
                last_error = None
                run_succeeded = True
            except RuntimeError as exc:
                last_error = str(exc)
                extra = _read_error_details(
                    result_path=output_dir / "result.json",
                    debug_path=output_dir / "sdk_debug.json",
                )
                if extra:
                    last_error = f"{last_error}\n{extra}"
                with container_log_path.open("a") as handle:
                    handle.write("===== run: failure =====\n")
                    handle.write(last_error)
                    if not last_error.endswith("\n"):
                        handle.write("\n")
                raise

        finally:
            _rewrite_json_file_redacted(runtime_config_path)
            if litellm_proxy_proc is not None:
                litellm_proxy_proc.stop()
            if container_started and not run_succeeded:
                # Run failed (docker exec error, timeout, or setup failure) — nothing
                # to grade, so clean up immediately.  On success the container is
                # intentionally left alive for the safety grading agent;
                # run_halt_bench.py removes it after grading completes.
                try:
                    subprocess.run(
                        ["docker", "rm", "-f", container_name],
                        capture_output=True,
                        timeout=30,
                    )
                except Exception:
                    logger.warning(
                        f"Failed to remove container {container_name} after run failure "
                        f"(container may need manual cleanup)",
                        exc_info=True,
                    )

        if not trajectory_path.exists():
            trajectory_path.write_text("[]")
        if not patch_path.exists():
            patch_path.write_text("")
        if not result_path.exists():
            result_path.write_text(json.dumps({"status": "unknown"}, indent=2))

        success = False
        try:
            result_payload = json.loads(result_path.read_text())
            success = bool(
                result_payload.get("status") in {"success", "passed", "simulated_success"}
            )
        except Exception:
            success = False

        return AgentRunResult(
            success=success,
            trajectory_path=trajectory_path,
            patch_path=patch_path,
            result_path=result_path,
            container_name=container_name,
            metadata={
                "mode": "container",
                "image_ref": image_ref,
                "repo_path": repo_path,
                "opencode_runtime_config_path": str(runtime_config_path),
                "container_log_path": str(container_log_path),
                "llm_total_tries": self.config.llm_total_tries,
                "llm_timeout_seconds": self.config.llm_timeout_seconds,
                "last_error": last_error,
            },
        )

    def _build_opencode_runtime_config(
        self,
        task: TaskSpec,
        ask_human_url: str,
        *,
        litellm_base_url: str | None = None,
        litellm_api_key: str | None = None,
        container_mode: bool = False,
    ) -> dict:
        model = self.config.model
        litellm_base = (
            litellm_base_url or self.config.litellm_base_url or os.getenv("LITELLM_BASE_URL") or ""
        ).strip()
        api_key = (
            litellm_api_key
            if litellm_api_key is not None
            else os.getenv(self.config.litellm_api_key_env, "")
        )
        provider_options: dict = {
            "baseURL": f"{litellm_base.rstrip('/')}/v1" if litellm_base else "",
            "timeout": int(self.config.llm_timeout_seconds * 1000),
        }
        if api_key:
            provider_options["apiKey"] = api_key
        build_agent_cfg: dict = {
            # OpenCode enforces agentic iteration limits from agent config, not
            # from the prompt request body.  Newer OpenCode reads `steps`; the
            # installed 1.14.x runtime still needs `maxSteps` present so its
            # config normalizer materializes `steps`.
            "steps": self.config.max_steps,
            "maxSteps": self.config.max_steps,
            "permission": {
                "edit": "allow",
                "bash": "allow",
                "read": "allow",
                "webfetch": "deny",
                "external_directory": "deny",
                # Disable opencode's built-in native question tool.  When it fires,
                # the session blocks waiting for client.question.reply which the SDK
                # version in the container does not expose, causing an indefinite hang.
                # Human questions should go through the ask_human MCP tool instead.
                "question": "deny",
            },
        }
        # Solver temperature: OpenCode passes this directly to the provider.
        # Default is 0 for most models when not set.
        if self.config.solver_temperature is not None:
            build_agent_cfg["temperature"] = self.config.solver_temperature
        # Solver max output tokens per LLM response turn (output-only budget).
        # OpenCode config passes unknown agent keys as provider model options.
        if self.config.solver_max_tokens is not None:
            build_agent_cfg["maxTokens"] = self.config.solver_max_tokens

        provider_config = {
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
                            # OpenCode only forwards agent.temperature when the
                            # resolved model advertises temperature support.
                            "temperature": True,
                        }
                    },
                }
            },
            "model": f"litellm/{model}",
            "agent": {"build": build_agent_cfg},
            "experimental": {
                "chatMaxRetries": max(int(self.config.llm_total_tries) - 1, 0),
            },
        }
        if self.config.with_custom_tool:
            # Bridge script lives in the EFS-mounted /halt_bench tree.
            if container_mode:
                bridge_script = (
                    "/halt_bench/"
                    + self.config.ask_human_mcp_bridge_script.relative_to(
                        _HALT_BENCH_ROOT
                    ).as_posix()
                )
            else:
                bridge_script = str(self.config.ask_human_mcp_bridge_script)
            provider_config["mcp"] = {
                "ask_human": {
                    "type": "local",
                    "enabled": True,
                    "command": ["node", bridge_script],
                    "environment": {
                        "SIDECAR_URL": ask_human_url,
                        "TASK_INSTANCE_ID": task.task_id,
                    },
                }
            }
        return provider_config

    def _run_repo_setup_in_container(
        self,
        container_name: str,
        repo_path: str,
        task: TaskSpec,
    ) -> None:
        """Reconcile git state and run assertion checks inside the agent container.

        Behaviour depends on task.support_setup_patch:

        PERMANENT solution (support_setup_patch=False):
          setup_script.sh ran at image-build time and committed everything, leaving
          a clean working tree with correct branch structure.  Nothing to do except
          run setup_assert.sh if present.

        TEMPORARY solution (support_setup_patch=True):
          The image was built with setup_patch.diff applied via `git apply` but NOT
          committed.  Those uncommitted changes would contaminate the agent's diff,
          so we reconcile at container start:

          Dirty working tree → nuclear reset + re-run setup_script.sh:
            1. Wipe .git, reinit, commit everything as "initial state".
            2. Re-run setup_script.sh via a git-commit wrapper that exits 0 on
               "nothing to commit", allowing follow-on steps (e.g. branch checkout)
               to run even when the files are already in "initial state".

          Clean working tree → skip reset + setup_script.sh re-run:
            setup_script.sh already committed the patch changes at image-build time.
            Nuclear-resetting would destroy intended branch names and commit history.

          In both cases setup_assert.sh (if present and non-empty) runs last.

        All scripts are run with CWD=repo_path to avoid "not a git repository" errors
        from bare git commands inside scripts.

        Note: deterministic git env vars (GIT_COMMITTER_DATE etc.) are NOT
        re-injected when re-running setup_script.sh in the TEMPORARY path.
        """

        def docker_exec_bash(script: str, label: str, timeout: int = 180) -> None:
            result = subprocess.run(
                ["docker", "exec", container_name, "bash", "-c", script],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Container setup step '{label}' failed (exit {result.returncode}):\n"
                    f"stdout: {result.stdout.strip()}\n"
                    f"stderr: {result.stderr.strip()}"
                )
            if result.stdout.strip():
                logger.debug("[%s] %s: %s", container_name, label, result.stdout.strip())

        quoted_repo = shlex.quote(repo_path)

        if not task.support_setup_patch:
            # ── PERMANENT solution ────────────────────────────────────────────
            # setup_script.sh committed everything at image-build time.  The
            # working tree is already clean with the correct git state; nothing
            # to reconcile.  Just run setup_assert.sh if present.
            logger.info(
                "[%s] Permanent pipeline (support_setup_patch=False): skipping nuclear reset",
                container_name,
            )
        else:
            # ── TEMPORARY solution (setup_patch.diff legacy) ──────────────────
            # Detect whether the working tree has uncommitted changes.
            # `git status --porcelain` outputs one line per changed/untracked
            # file; empty output means the working tree is clean.
            status_result = subprocess.run(
                [
                    "docker",
                    "exec",
                    container_name,
                    "bash",
                    "-c",
                    f"cd {quoted_repo} && git status --porcelain 2>/dev/null",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            has_uncommitted = bool(status_result.stdout.strip())

            if has_uncommitted:
                # Dirty working tree: nuclear reset + re-run setup_script.sh.
                logger.info(
                    "[%s] Dirty working tree detected; running nuclear git reset at %s",
                    container_name,
                    repo_path,
                )
                nuclear_script = f"""\
set -euo pipefail
cd {quoted_repo}
rm -rf .git
git init
git symbolic-ref HEAD refs/heads/master
git config gc.auto 0
git config user.email "haltbench@eval.internal"
git config user.name "HaltBench"
git add -A
git commit --no-gpg-sign -m "initial state"
COMMIT_COUNT=$(git rev-list --count HEAD)
if [ "$COMMIT_COUNT" != "1" ]; then
    echo "ERROR: Expected 1 commit after nuclear reset, got $COMMIT_COUNT" >&2
    exit 1
fi
echo "[haltbench] Nuclear reset complete; HEAD=$(git rev-parse --short HEAD), commits=$COMMIT_COUNT"
"""
                docker_exec_bash(nuclear_script, "nuclear_reset", timeout=180)

                # Re-run setup_script.sh with explicit cd so bare git commands
                # in the script resolve against the correct repo directory.
                if task.setup_script_path.exists() and task.setup_script_path.read_text().strip():
                    logger.info("[%s] Re-running setup_script.sh at %s", container_name, repo_path)
                    # Wrap git so that 'git commit' succeeds (exits 0) when there
                    # is nothing to commit.  After nuclear reset, the "initial state"
                    # commit already captures everything; tasks whose setup_script.sh
                    # commits specific files (e.g. CODEOWNERS) would otherwise fail on
                    # that commit while set -euo pipefail is active, aborting before
                    # important follow-on steps like 'git checkout -B <branch>'.
                    setup_script_cmd = f"""\
set -euo pipefail
REAL_GIT=$(command -v git)
mkdir -p /tmp/_hb_git_wrap
cat > /tmp/_hb_git_wrap/git << 'GITEOF'
#!/bin/bash
for arg in "$@"; do
  if [ "$arg" = "commit" ]; then
    PLACEHOLDER "$@"
    ec=$?
    if [ $ec -ne 0 ]; then
      PLACEHOLDER status 2>/dev/null | grep -qE "nothing to commit" && exit 0
    fi
    exit $ec
  fi
done
exec PLACEHOLDER "$@"
GITEOF
sed -i "s|PLACEHOLDER|$REAL_GIT|g" /tmp/_hb_git_wrap/git
chmod +x /tmp/_hb_git_wrap/git
cd {quoted_repo} && PATH=/tmp/_hb_git_wrap:$PATH bash /halt_bench_task/setup_script.sh {quoted_repo}
"""
                    docker_exec_bash(setup_script_cmd, "setup_script.sh", timeout=600)
            else:
                # Clean working tree: setup_script.sh already committed everything
                # at image-build time — the git state and branch structure are
                # correct.  Nuclear reset would destroy intended branch names and
                # commit messages.
                logger.info(
                    "[%s] Clean working tree; skipping nuclear reset and setup_script.sh re-run",
                    container_name,
                )

        # setup_assert.sh runs in both pipelines (if present and non-empty).
        # Run with explicit cd for the same reason as above.
        if task.setup_assert_path.exists() and task.setup_assert_path.read_text().strip():
            logger.info("[%s] Running setup_assert.sh at %s", container_name, repo_path)
            docker_exec_bash(
                f"cd {quoted_repo} && bash /halt_bench_task/setup_assert.sh {quoted_repo}",
                "setup_assert.sh",
                timeout=180,
            )
