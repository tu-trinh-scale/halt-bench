from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import logging
import os
import subprocess
import sys
import threading
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

from halt_bench.agents.registry import build_agent
from halt_bench.core.blockers import BlockerEntry, BlockerRegistry
from halt_bench.core.tasks import TaskSpec
from halt_bench.evaluation.runner import evaluate_task_run
from halt_bench.evaluation.schema import EvaluationResult
from halt_bench.runtime.ask_human_process import start_ask_human_process
from halt_bench.runtime.env import load_project_env
from halt_bench.runtime.litellm_direct_process import (
    LiteLLMDirectProcess,
    start_litellm_direct_proxy,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent
_TASKS_DIR = _PROJECT_ROOT / "tasks"
_OUTPUTS_DIR = _PROJECT_ROOT / "outputs"

_CSV_LOCK = threading.Lock()
_CSV_FIELDNAMES = [
    "model",
    "mode",
    "instance_id",
    "task_completed",
    "harm_execution_rate",
    "safety_grading_reasoning",
    "harm_discovery_rate",
    "question_efficiency_rate",
    "num_questions",
    "risk_verbalization_score",
    "risk_verbalization_reasoning",
    "risk_disclosure_score",
    "risk_disclosure_reasoning",
    "eval_awareness_score",
    "eval_awareness_reasoning",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_full_info_user_request(user_request: str, blockers: list[BlockerEntry]) -> str:
    """Return the user request augmented with all blocker resolutions."""
    if not blockers:
        return user_request
    clarifications = "\n\n# Additional Clarifications"
    for blocker in blockers:
        clarifications += f"\n## {blocker.description}\n{blocker.resolution}"
    return user_request + clarifications


def _resolve_ask_guidance_path(guidance_name: str | None) -> Path | None:
    if not guidance_name:
        return None
    ask_guidance_dir = _PROJECT_ROOT / "halt_bench" / "agents" / "ask_guidances"
    requested = Path(guidance_name.strip()).name
    if not requested.endswith(".md"):
        requested = f"{requested}.md"
    candidate = ask_guidance_dir / requested
    if candidate.is_file():
        return candidate.resolve()
    available = ", ".join(
        sorted(p.stem for p in ask_guidance_dir.glob("*") if p.is_file())
    )
    raise FileNotFoundError(
        f"Unknown ask guidance {guidance_name!r}. "
        f"Available: {available or '[none found]'}"
    )


def _ensure_image(task: TaskSpec) -> None:
    """Build the task image if it does not yet exist locally."""
    result = subprocess.run(
        ["docker", "image", "inspect", task.image_tag],
        capture_output=True,
    )
    if result.returncode == 0:
        return
    logger.info(
        "Image %r not found locally — attempting to build it now.", task.image_tag
    )
    build_script = _PROJECT_ROOT / "build_halt_bench_images.py"
    subprocess.run(
        [sys.executable, str(build_script), "--instance-id", task.instance_id],
        check=True,
    )


def _append_results_csv(
    results_csv: Path,
    *,
    model: str,
    mode: str,
    instance_id: str,
    task_completed: bool,
    harm_execution_rate: float | None,
    safety_grading_reasoning: str,
    harm_discovery_rate: float | None,
    question_efficiency_rate: float | None,
    num_questions: int | None,
    risk_verbalization_score: float | None,
    risk_verbalization_reasoning: str | None,
    risk_disclosure_score: float | None,
    risk_disclosure_reasoning: str | None,
    eval_awareness_score: float | None,
    eval_awareness_reasoning: str | None,
) -> None:
    """Append one row to outputs/<run_id>/results.csv."""
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not results_csv.exists()
    row = {
        "model": model,
        "mode": mode,
        "instance_id": instance_id,
        "task_completed": task_completed,
        "harm_execution_rate": harm_execution_rate,
        "safety_grading_reasoning": safety_grading_reasoning,
        "harm_discovery_rate": harm_discovery_rate,
        "question_efficiency_rate": question_efficiency_rate,
        "num_questions": num_questions,
        "risk_verbalization_score": risk_verbalization_score,
        "risk_verbalization_reasoning": risk_verbalization_reasoning,
        "risk_disclosure_score": risk_disclosure_score,
        "risk_disclosure_reasoning": risk_disclosure_reasoning,
        "eval_awareness_score": eval_awareness_score,
        "eval_awareness_reasoning": eval_awareness_reasoning,
    }
    with _CSV_LOCK:
        with results_csv.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDNAMES)
            if write_header:
                writer.writeheader()
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Core run logic
# ---------------------------------------------------------------------------


def run_task(
    task: TaskSpec,
    *,
    model: str = "claude-opus-4-6",
    mode: str = "ask_human",
    max_steps: int = 200,
    solver_backend: str = "proxy",
    grader_backend: str = "proxy",
    llm_total_tries: int = 3,
    llm_timeout_seconds: int = 300,
    solve_timeout_seconds: int = 7200,
    eval_timeout_seconds: int = 3600,
    with_custom_tool: bool = False,
    with_ask_guidance: str | None = None,
    enable_meta_eval: bool = False,
    safety_grading_model: str = "gemini/gemini-3.1-flash-lite",
    safety_grading_timeout_seconds: int = 1800,
    simulate: bool = False,
    ask_human_port: int = 0,
    ask_human_backend: str = "vllm",
    run_output_dir: Path | None = None,
    solver_temperature: float | None = None,
    solver_max_tokens: int | None = None,
) -> dict:
    """Run one task/model/mode and return a result dict."""
    if mode == "full_info" and with_ask_guidance:
        raise ValueError("--with-ask-guidance is prohibited in full_info mode")

    ask_guidance_path = _resolve_ask_guidance_path(with_ask_guidance)

    run_output_dir = run_output_dir or (_OUTPUTS_DIR / "_unsaved" / uuid.uuid4().hex[:8])
    run_output_dir.mkdir(parents=True, exist_ok=True)

    # ── Solver backend ────────────────────────────────────────────────────────
    # proxy:  drop-params proxy (local Node.js) → LITELLM_BASE_URL (remote proxy)
    # direct: locally-spawned LiteLLM proxy server → provider API key
    solver_direct_proc: LiteLLMDirectProcess | None = None
    if solver_backend == "direct":
        solver_direct_proc = start_litellm_direct_proxy(model=model)
        effective_solver_litellm_url = solver_direct_proc.url
        solver_use_drop_params_proxy = False
    else:
        effective_solver_litellm_url = os.getenv("LITELLM_BASE_URL", "")
        solver_use_drop_params_proxy = True

    # ── Grader backend ────────────────────────────────────────────────────────
    grader_direct_proc: LiteLLMDirectProcess | None = None
    if grader_backend == "direct":
        grader_direct_proc = start_litellm_direct_proxy(model=safety_grading_model)
        effective_grader_litellm_url = grader_direct_proc.url
        grader_use_drop_params_proxy = False
    else:
        effective_grader_litellm_url = os.getenv("LITELLM_BASE_URL", "")
        grader_use_drop_params_proxy = True

    # ── Ask-human backend ─────────────────────────────────────────────────────
    # vllm:  VLLM_BASE_URL (local vLLM inference server)
    # proxy: LITELLM_BASE_URL + LITELLM_API_KEY (remote LiteLLM proxy)
    _ask_provider = "litellm_proxy" if ask_human_backend == "proxy" else "vllm"
    ask_process = start_ask_human_process(
        tasks_dir=task.task_dir,
        port=ask_human_port,
        provider=_ask_provider,
    )

    solver_container: str | None = None
    eval_result: EvaluationResult | None = None

    try:
        effective_with_custom_tool = with_custom_tool and mode != "full_info"
        agent = build_agent(
            "opencode",
            simulate=simulate,
            model=model,
            max_steps=max_steps,
            litellm_base_url=effective_solver_litellm_url or None,
            use_local_drop_params_proxy=solver_use_drop_params_proxy,
            llm_total_tries=llm_total_tries,
            llm_timeout_seconds=llm_timeout_seconds,
            run_timeout_seconds=solve_timeout_seconds,
            with_custom_tool=effective_with_custom_tool,
            with_ask_guidance=ask_guidance_path is not None,
            ask_guidance_path=ask_guidance_path,
            solver_temperature=solver_temperature,
            solver_max_tokens=solver_max_tokens,
        )

        run_subdir = run_output_dir / "run"
        run_subdir.mkdir(parents=True, exist_ok=True)

        user_request_override_path: Path | None = None
        if mode == "full_info":
            registry = BlockerRegistry.model_validate(
                json.loads(task.blocker_registry_path.read_text())
            )
            full_info_text = _build_full_info_user_request(
                task.user_request_path.read_text(), registry.blockers
            )
            user_request_override_path = run_subdir / "full_info_user_request.md"
            user_request_override_path.write_text(full_info_text)

        run_result = agent.run(
            task,
            output_dir=run_subdir,
            ask_human_url=ask_process.url,
            user_request_override_path=user_request_override_path,
        )
        solver_container = run_result.container_name

        # Fetch ask_human logs.
        ask_human_logs: dict = {}
        try:
            req = urllib.request.Request(
                f"{ask_process.url}/get_logs",
                data=b"{}",
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                ask_human_logs = json.loads(resp.read().decode())
        except Exception:
            pass

        ask_human_logs_path = run_output_dir / "ask_human_logs.json"
        ask_human_logs_path.write_text(json.dumps(ask_human_logs, indent=2))

        run_meta: dict = {}
        if run_result.result_path.exists():
            try:
                run_meta = json.loads(run_result.result_path.read_text())
            except Exception:
                pass

        trajectory_data: list[dict] = []
        try:
            if run_result.trajectory_path.exists():
                trajectory_data = json.loads(run_result.trajectory_path.read_text())
        except Exception:
            pass

        agent_patch_text: str = ""
        try:
            if run_result.patch_path is not None and run_result.patch_path.exists():
                text = run_result.patch_path.read_text()
                if text.strip():
                    agent_patch_text = text
        except Exception:
            pass

        eval_path = run_output_dir / "evaluation.json"
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    evaluate_task_run,
                    task_dir=task.task_dir,
                    trajectory_path=run_result.trajectory_path,
                    patch_path=run_result.patch_path,
                    ask_human_logs_path=ask_human_logs_path,
                    mode=mode,
                    enable_meta_eval=enable_meta_eval,
                    output_path=eval_path,
                    task=task,
                    run_output_dir=run_subdir,
                    container_name=solver_container,
                    safety_grading_model=safety_grading_model,
                    litellm_base_url=effective_grader_litellm_url or None,
                    use_local_drop_params_proxy=grader_use_drop_params_proxy,
                    safety_grading_timeout_seconds=safety_grading_timeout_seconds,
                    run_tests=True,
                )
                eval_result = future.result(timeout=float(eval_timeout_seconds))
        finally:
            if solver_container:
                subprocess.run(
                    ["docker", "rm", "-f", solver_container],
                    capture_output=True,
                    timeout=30,
                )

        return {
            "instance_id": task.instance_id,
            "mode": mode,
            "model": model,
            "run_success": run_result.success,
            "run_error": run_meta.get("error"),
            "num_steps": run_meta.get("num_steps", 0),
            "evaluation": eval_result.model_dump() if eval_result else None,
            "saved_to": str(run_output_dir),
            "trajectory": trajectory_data,
            "agent_patch": agent_patch_text,
        }

    finally:
        ask_process.stop()
        if solver_direct_proc is not None:
            solver_direct_proc.stop()
        if grader_direct_proc is not None:
            grader_direct_proc.stop()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    load_project_env(_PROJECT_ROOT)

    parser = argparse.ArgumentParser(
        description="Run HALT-Bench tasks (one or many, optionally in parallel).",
    )

    # Task selection — at least one of these is required.
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument(
        "--instance-id",
        nargs="+",
        metavar="ID",
        help="One or more task instance IDs (folder names under tasks/).",
    )
    task_group.add_argument(
        "--all-tasks",
        action="store_true",
        help="Run every task folder found under tasks/.",
    )

    parser.add_argument("--model", default="claude-opus-4-6")
    parser.add_argument("--mode", choices=["ask_human", "full_info"], default="ask_human")
    parser.add_argument(
        "--run-id",
        metavar="ID",
        default=None,
        help="Run identifier used for the output directory (outputs/<run-id>/). "
             "Auto-generated from timestamp if not provided.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=5,
        metavar="N",
        help="Maximum number of tasks to run in parallel (default: 5).",
    )
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument(
        "--solver-backend",
        choices=["proxy", "direct"],
        default="proxy",
        help="LLM backend for the solver agent. "
             "'proxy' uses LITELLM_BASE_URL + LITELLM_API_KEY (default). "
             "'direct' calls the provider directly using provider API keys "
             "(e.g. ANTHROPIC_API_KEY, OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--grader-backend",
        choices=["proxy", "direct"],
        default="proxy",
        help="LLM backend for the safety grader and meta-eval. "
             "'proxy' uses LITELLM_BASE_URL + LITELLM_API_KEY (default). "
             "'direct' calls the provider directly using provider API keys.",
    )
    parser.add_argument(
        "--ask-human-backend",
        choices=["vllm", "proxy"],
        default="vllm",
        help="Backend for the ask_human oracle. "
             "'vllm' uses VLLM_BASE_URL (default). "
             "'proxy' uses LITELLM_BASE_URL + LITELLM_API_KEY.",
    )
    parser.add_argument("--opencode-llm-total-tries", type=int, default=3)
    parser.add_argument("--opencode-llm-timeout-seconds", type=int, default=300)
    parser.add_argument("--solve-timeout-seconds", type=int, default=7200)
    parser.add_argument("--eval-timeout-seconds", type=int, default=3600)
    parser.add_argument("--with-custom-tool", action="store_true")
    parser.add_argument(
        "--with-ask-guidance",
        metavar="GUIDANCE_NAME",
        help="Ask-guidance file name from halt_bench/agents/ask_guidances/.",
    )
    parser.add_argument("--simulate-agent", action="store_true")
    parser.add_argument("--enable-meta-eval", action="store_true")
    parser.add_argument("--ask-human-port", type=int, default=0)
    parser.add_argument(
        "--safety-grading-model",
        default="gemini/gemini-3.1-flash-lite",
        metavar="MODEL",
    )
    parser.add_argument("--safety-grading-timeout-seconds", type=int, default=1800)
    parser.add_argument("--solver-temperature", type=float, default=None)
    parser.add_argument("--solver-max-tokens", type=int, default=None)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose proxy logging. Proxy request/response details are written "
             "to a temp file (printed to stderr at startup).",
    )
    parser.add_argument(
        "--results-in-task", "-rit",
        action="store_true",
        help="Save trajectory and metrics directly into the task folder as "
             "<model>_trajectory.json and <model>_results.json. "
             "Model name is derived from --model (slash prefix stripped, hyphens→underscores).",
    )
    args = parser.parse_args()

    if args.mode == "full_info" and args.with_ask_guidance:
        parser.error("--with-ask-guidance is prohibited in full_info mode")

    # Resolve the list of instance IDs to run.
    if args.all_tasks:
        instance_ids = sorted(d.name for d in _TASKS_DIR.iterdir() if d.is_dir())
        if not instance_ids:
            print(f"Error: no task folders found under {_TASKS_DIR}", file=sys.stderr)
            sys.exit(1)
    else:
        instance_ids = args.instance_id

    # Resolve run_id.  Every task in this invocation shares the same run_id so
    # their outputs are grouped under outputs/<run_id>/<instance_id>/.
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_root = _OUTPUTS_DIR / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    if args.debug:
        proxy_log = run_root / "proxy_debug.log"
        os.environ["RESPONSES_PROXY_LOG_FILE"] = str(proxy_log)
        logger.info("Debug mode enabled — proxy logs → %s", proxy_log)

    rit_prefix: str | None = None
    if args.results_in_task:
        raw = args.model
        if "/" in raw:
            raw = raw.split("/", 1)[1]
        rit_prefix = raw.replace("-", "_")

    logger.info(
        "Run ID: %s  →  %s  |  tasks: %d  |  concurrency: %d",
        run_id, run_root, len(instance_ids), args.max_concurrency,
    )

    # Shared keyword args for run_task (everything except the task itself and
    # the per-task output dir).
    run_kwargs: dict = dict(
        model=args.model,
        mode=args.mode,
        max_steps=args.max_steps,
        solver_backend=args.solver_backend,
        grader_backend=args.grader_backend,
        ask_human_backend=args.ask_human_backend,
        llm_total_tries=args.opencode_llm_total_tries,
        llm_timeout_seconds=args.opencode_llm_timeout_seconds,
        solve_timeout_seconds=args.solve_timeout_seconds,
        eval_timeout_seconds=args.eval_timeout_seconds,
        with_custom_tool=args.with_custom_tool,
        with_ask_guidance=args.with_ask_guidance,
        enable_meta_eval=args.enable_meta_eval,
        safety_grading_model=args.safety_grading_model,
        safety_grading_timeout_seconds=args.safety_grading_timeout_seconds,
        simulate=args.simulate_agent,
        ask_human_port=args.ask_human_port,
        solver_temperature=args.solver_temperature,
        solver_max_tokens=args.solver_max_tokens,
    )

    def _run_one(instance_id: str) -> dict:
        task_dir = _TASKS_DIR / instance_id
        if not task_dir.is_dir():
            logger.error("Task folder not found: %s", task_dir)
            return {"instance_id": instance_id, "error": "task folder not found"}
        task = TaskSpec.from_task_dir(task_dir)
        _ensure_image(task)
        per_task_dir = run_root / instance_id
        per_task_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = run_task(task, **run_kwargs, run_output_dir=per_task_dir)
        except Exception as exc:
            logger.exception("Task %s failed with exception: %s", instance_id, exc)
            result = {"instance_id": instance_id, "error": str(exc), "evaluation": None}

        eval_data = result.get("evaluation") or {}
        tc = eval_data.get("test_correctness") or {}
        if not rit_prefix:
            # Append to per-run results CSV (thread-safe via _CSV_LOCK inside).
            _append_results_csv(
                run_root / "results.csv",
                model=args.model,
                mode=args.mode,
                instance_id=instance_id,
                task_completed=bool(tc.get("passed", False)),
                harm_execution_rate=eval_data.get("harm_execution_rate"),
                safety_grading_reasoning=eval_data.get("safety_grading_reasoning", ""),
                harm_discovery_rate=eval_data.get("harm_discovery_rate"),
                question_efficiency_rate=eval_data.get("question_efficiency_rate"),
                num_questions=eval_data.get("num_questions"),
                risk_verbalization_score=eval_data.get("risk_verbalization_score"),
                risk_verbalization_reasoning=eval_data.get("risk_verbalization_reasoning"),
                risk_disclosure_score=eval_data.get("risk_disclosure_score"),
                risk_disclosure_reasoning=eval_data.get("risk_disclosure_reasoning"),
                eval_awareness_score=eval_data.get("eval_awareness_score"),
                eval_awareness_reasoning=eval_data.get("eval_awareness_reasoning"),
            )
        if rit_prefix:
            metrics = {
                "task_completed": bool(tc.get("passed", False)),
                "harm_execution_rate": eval_data.get("harm_execution_rate"),
                "safety_grading_reasoning": eval_data.get("safety_grading_reasoning", ""),
                "harm_discovery_rate": eval_data.get("harm_discovery_rate"),
                "question_efficiency_rate": eval_data.get("question_efficiency_rate"),
                "num_questions": eval_data.get("num_questions"),
                "risk_verbalization_score": eval_data.get("risk_verbalization_score"),
                "risk_verbalization_reasoning": eval_data.get("risk_verbalization_reasoning"),
                "risk_disclosure_score": eval_data.get("risk_disclosure_score"),
                "risk_disclosure_reasoning": eval_data.get("risk_disclosure_reasoning"),
                "eval_awareness_score": eval_data.get("eval_awareness_score"),
                "eval_awareness_reasoning": eval_data.get("eval_awareness_reasoning"),
            }
            task_dir = _TASKS_DIR / instance_id
            (task_dir / f"{rit_prefix}_results.json").write_text(
                json.dumps(metrics, indent=2, default=str) + "\n"
            )
            trajectory = result.get("trajectory") or []
            (task_dir / f"{rit_prefix}_trajectory.json").write_text(
                json.dumps(trajectory, indent=2, default=str) + "\n"
            )
            logger.info(
                "Task %s: saved %s_results.json and %s_trajectory.json to task folder",
                instance_id, rit_prefix, rit_prefix,
            )

        logger.info("Task %s done → %s", instance_id, per_task_dir)
        return result

    results: list[dict] = []
    if len(instance_ids) == 1:
        results.append(_run_one(instance_ids[0]))
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.max_concurrency
        ) as pool:
            futures = {pool.submit(_run_one, iid): iid for iid in instance_ids}
            for future in concurrent.futures.as_completed(futures):
                iid = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    logger.error("Unhandled exception for task %s: %s", iid, exc)
                    results.append({"instance_id": iid, "error": str(exc)})

    if not rit_prefix:
        logger.info("Results appended to %s", run_root / "results.csv")
    _OMIT = {"trajectory", "agent_patch"}
    printable = [{k: v for k, v in r.items() if k not in _OMIT} for r in results]
    print(json.dumps(printable if len(printable) > 1 else printable[0], indent=2, default=str))
    logger.info("All done. Run artifacts in %s", run_root)


if __name__ == "__main__":
    main()
