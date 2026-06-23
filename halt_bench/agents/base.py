from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from halt_bench.core.tasks import TaskSpec
from pydantic import BaseModel, Field


class AgentRunResult(BaseModel):
    success: bool
    trajectory_path: Path
    patch_path: Path
    result_path: Path
    # The running Docker container is kept alive after the solver finishes so
    # the safety grader can run inside the exact same environment.  The caller
    # (run_halt_bench.py) is responsible for `docker rm -f` after grading.
    # None when running in simulate mode or when no container was started.
    container_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class HaltAgent(ABC):
    @abstractmethod
    def run(
        self,
        task: TaskSpec,
        *,
        output_dir: Path,
        ask_human_url: str,
        user_request_override_path: Path | None = None,
    ) -> AgentRunResult:
        """Run an agent on a task artifact.

        Args:
            task: The task to run.
            output_dir: Directory for all run artifacts (trajectory, patch, logs, etc.).
            ask_human_url: URL of the ask_human sidecar server.
            user_request_override_path: If set, the agent uses this file as the user request
                instead of the default task user_request.md (used in full_info mode).
        """
