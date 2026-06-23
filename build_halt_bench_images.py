from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from halt_bench.core.tasks import TaskSpec, make_image_tag
from halt_bench.runtime.command import run_command
from halt_bench.runtime.docker_auth import get_dockerhub_env_vars
from halt_bench.runtime.env import load_project_env

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent
_TASKS_DIR = _PROJECT_ROOT / "tasks"
_DOCKERFILE = _PROJECT_ROOT / "docker" / "Dockerfile.task_runtime"
_REPO_PATH_IN_IMAGE = "/app"


# ---------------------------------------------------------------------------
# Patch normalization
# ---------------------------------------------------------------------------


def _normalize_patch_line_endings(patch_content: str) -> str:
    """Strip \\r from patch files before applying them in a Linux container."""
    if not patch_content or not patch_content.strip():
        return patch_content
    lines = patch_content.split("\n")
    result = []
    for line in lines:
        if (
            line.startswith("diff --git ")
            or line.startswith("--- ")
            or line.startswith("+++ ")
            or line.startswith("index ")
            or line.startswith("@@ ")
            or line.startswith((" ", "+", "-"))
        ):
            line = line.rstrip("\r")
        result.append(line)
    return "\n".join(result)


# ---------------------------------------------------------------------------
# Docker helpers
# ---------------------------------------------------------------------------


def _image_exists(image_tag: str) -> bool:
    """Return True if a Docker image with the given tag exists locally."""
    result = subprocess.run(
        ["docker", "image", "inspect", image_tag],
        capture_output=True,
    )
    return result.returncode == 0


def _docker_login_if_available() -> None:
    creds = get_dockerhub_env_vars()
    username = creds.get("DOCKERHUB_USERNAME", "")
    token = creds.get("DOCKERHUB_TOKEN", "")
    if not username or not token:
        return
    run_command(["docker", "login", "-u", username, "-p", token], check=False)


def _get_seed_runtime_config(seed_image: str) -> dict[str, Any]:
    res = run_command(
        ["docker", "inspect", "--format", "{{json .Config}}", seed_image],
        check=False,
    )
    if res.returncode != 0 or not res.stdout.strip():
        return {}
    try:
        parsed = json.loads(res.stdout.strip())
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _runtime_restore_changes(seed_runtime: dict[str, Any]) -> list[str]:
    """Build --change flags for docker commit to restore original Entrypoint/Cmd."""
    changes: list[str] = []
    entrypoint = seed_runtime.get("Entrypoint")
    cmd = seed_runtime.get("Cmd")
    if entrypoint is None:
        changes.extend(["--change", "ENTRYPOINT []"])
    elif isinstance(entrypoint, list):
        changes.extend(["--change", f"ENTRYPOINT {json.dumps(entrypoint)}"])
    if cmd is None:
        changes.extend(["--change", "CMD []"])
    elif isinstance(cmd, list):
        changes.extend(["--change", f"CMD {json.dumps(cmd)}"])
    return changes


def _create_keepalive_container(seed_image: str) -> str:
    for shell in ("/bin/sh", "/bin/bash", "sh", "bash"):
        res = run_command(
            [
                "docker",
                "create",
                "--label",
                f"haltbench_owner_pid={os.getpid()}",
                "--entrypoint",
                shell,
                seed_image,
                "-lc",
                "while true; do sleep 3600; done",
            ],
            check=False,
        )
        if res.returncode == 0:
            container_id = res.stdout.strip()
            if container_id:
                return container_id
    raise RuntimeError("Failed to create keepalive build container with sh/bash entrypoint")


def _get_volume_paths(seed_image: str) -> list[str]:
    res = run_command(
        ["docker", "inspect", "--format", "{{json .Config.Volumes}}", seed_image],
        check=False,
    )
    if res.returncode != 0 or not res.stdout.strip() or res.stdout.strip() == "null":
        return []
    try:
        return list(json.loads(res.stdout.strip()).keys())
    except Exception:
        return []


def _snapshot_volumes(container_id: str, volume_paths: list[str]) -> dict[str, bool]:
    snapshot: dict[str, bool] = {}
    for path in volume_paths:
        res = run_command(
            ["docker", "exec", container_id, "find", path, "-type", "f",
             "-exec", "stat", "-c", "%n:%s:%Y", "{}", "+"],
            check=False,
        )
        if res.returncode == 0:
            for line in res.stdout.splitlines():
                if line.strip():
                    snapshot[line.strip()] = True
    return snapshot


# ---------------------------------------------------------------------------
# Core build logic
# ---------------------------------------------------------------------------


def _extract_seed_image(pull_command: str) -> str:
    """Extract the image reference from a 'docker pull <image>' string."""
    parts = pull_command.strip().split()
    if len(parts) >= 2 and parts[0] == "docker" and parts[1] == "pull":
        return parts[-1]
    raise ValueError(
        f"Cannot parse seed image from pull_command: {pull_command!r}. "
        "Expected format: 'docker pull <image>'"
    )


def build_task_image(task_dir: Path, *, force: bool = False) -> str:
    """Build the Docker image for a single task folder.

    Returns the image tag that was built (or skipped).
    Raises RuntimeError on build failure.
    """
    id_path = task_dir / "instance_id.json"
    if not id_path.exists():
        raise FileNotFoundError(f"instance_id.json not found in {task_dir}")

    id_data = json.loads(id_path.read_text())
    instance_id: str = id_data["instance_id"]
    image_tag = make_image_tag(instance_id)

    pull_path = task_dir / "pull_command.json"
    if not pull_path.exists():
        raise FileNotFoundError(f"pull_command.json not found in {task_dir}")
    pull_data = json.loads(pull_path.read_text())
    seed_image = _extract_seed_image(pull_data["pull_command"])

    # Skip if already built (unless --force).
    if not force and _image_exists(image_tag):
        logger.info("[%s] Image %r already exists — skipping.", instance_id, image_tag)
        return image_tag

    logger.info("[%s] Building image %r from seed %r", instance_id, image_tag, seed_image)

    setup_patch_path = task_dir / "setup_patch.diff"
    setup_script_path = task_dir / "setup_script.sh"
    setup_assert_path = task_dir / "setup_assert.sh"

    container_id: str | None = None
    temp_image = f"{image_tag}-base"

    try:
        _docker_login_if_available()
        run_command(["docker", "pull", seed_image])

        seed_runtime = _get_seed_runtime_config(seed_image)
        container_id = _create_keepalive_container(seed_image)

        run_command(["docker", "start", container_id])

        volume_paths = _get_volume_paths(seed_image)
        pre_snapshot = _snapshot_volumes(container_id, volume_paths)

        # Apply setup_patch.diff if non-empty.
        if setup_patch_path.exists():
            patch_text = _normalize_patch_line_endings(setup_patch_path.read_text())
            if patch_text.strip():
                run_command(
                    ["docker", "cp", str(setup_patch_path),
                     f"{container_id}:/tmp/haltbench_setup_patch.diff"]
                )
                run_command(
                    ["docker", "exec", container_id, "bash", "-lc",
                     f"cd {_REPO_PATH_IN_IMAGE} && git apply /tmp/haltbench_setup_patch.diff"]
                )

        # Run setup_script.sh if present and non-empty.
        if setup_script_path.exists() and setup_script_path.read_text().strip():
            run_command(
                ["docker", "cp", str(setup_script_path),
                 f"{container_id}:/tmp/haltbench_setup.sh"]
            )
            run_command(
                ["docker", "exec", container_id, "bash", "-lc",
                 f"chmod +x /tmp/haltbench_setup.sh && "
                 f"/tmp/haltbench_setup.sh {_REPO_PATH_IN_IMAGE}"]
            )

        # Check for unintended volume writes (warn only — don't abort).
        post_snapshot = _snapshot_volumes(container_id, volume_paths)
        if pre_snapshot != post_snapshot:
            diff_keys = set(pre_snapshot.keys()) ^ set(post_snapshot.keys())
            changed_paths = sorted({k.split(":", 1)[0] for k in diff_keys})
            logger.warning(
                "[%s] Volume writes detected on %s — data in these paths will NOT "
                "be committed to the image.",
                instance_id,
                changed_paths,
            )

        # Run setup_assert.sh if present and non-empty.
        if setup_assert_path.exists() and setup_assert_path.read_text().strip():
            run_command(
                ["docker", "cp", str(setup_assert_path),
                 f"{container_id}:/tmp/haltbench_setup_assert.sh"]
            )
            assert_res = run_command(
                ["docker", "exec", container_id, "bash", "-lc",
                 f"chmod +x /tmp/haltbench_setup_assert.sh && "
                 f"/tmp/haltbench_setup_assert.sh {_REPO_PATH_IN_IMAGE}"],
                check=False,
            )
            if assert_res.stdout:
                logger.info("[%s] Assertion stdout:\n%s", instance_id, assert_res.stdout)
            if assert_res.stderr:
                logger.info("[%s] Assertion stderr:\n%s", instance_id, assert_res.stderr)
            if assert_res.returncode != 0:
                raise AssertionError(
                    f"[{instance_id}] Build-time assertions failed "
                    f"(exit {assert_res.returncode}). Aborting commit."
                )

        # Clean up temp files from container.
        run_command(
            ["docker", "exec", container_id, "bash", "-lc",
             "rm -f /tmp/haltbench_setup_patch.diff /tmp/haltbench_setup.sh "
             "/tmp/haltbench_setup_assert.sh"],
            check=False,
        )

        # Commit patched container to a temporary base image.
        commit_cmd = ["docker", "commit", *_runtime_restore_changes(seed_runtime),
                      container_id, temp_image]
        run_command(commit_cmd)

        # Build final image: temp_image + Node 20 + npm deps via Dockerfile.task_runtime.
        try:
            run_command(
                ["docker", "build",
                 "--build-arg", f"BASE_IMAGE={temp_image}",
                 "-t", image_tag,
                 "-f", str(_DOCKERFILE),
                 "."],
                cwd=str(_PROJECT_ROOT),
            )
        finally:
            run_command(["docker", "rmi", temp_image], check=False)

        logger.info("[%s] Image %r built successfully.", instance_id, image_tag)
        return image_tag

    finally:
        if container_id:
            run_command(["docker", "rm", "-fv", container_id], check=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    load_project_env(_PROJECT_ROOT)

    parser = argparse.ArgumentParser(
        description="Build Docker images for HALT-Bench task folders."
    )
    parser.add_argument(
        "--instance-id",
        metavar="ID",
        help="Build only this specific task (folder name in tasks/). "
             "If omitted, builds all tasks in tasks/.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if the image already exists locally.",
    )
    args = parser.parse_args()

    if args.instance_id:
        task_dirs = [_TASKS_DIR / args.instance_id]
        if not task_dirs[0].is_dir():
            print(f"Error: task folder not found: {task_dirs[0]}", file=sys.stderr)
            sys.exit(1)
    else:
        task_dirs = sorted(
            p for p in _TASKS_DIR.iterdir()
            if p.is_dir() and (p / "instance_id.json").exists()
        )
        if not task_dirs:
            print(f"No task folders found in {_TASKS_DIR}", file=sys.stderr)
            sys.exit(1)

    errors: list[str] = []
    for task_dir in task_dirs:
        try:
            build_task_image(task_dir, force=args.force)
        except Exception as exc:
            logger.error("Failed to build %s: %s", task_dir.name, exc)
            errors.append(f"{task_dir.name}: {exc}")

    if errors:
        print(f"\n{len(errors)} task(s) failed:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
