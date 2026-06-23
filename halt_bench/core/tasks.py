from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, computed_field

_IMAGE_PREFIX = "official-halt-bench"
_DOCKER_TAG_MAX = 128


def make_image_tag(instance_id: str) -> str:
    """Return the Docker image tag for a given instance_id.

    Format: official-halt-bench:<instance_id>
    Truncated so the full tag (prefix + colon + instance_id) stays within
    Docker's 128-character tag limit.
    """
    prefix_and_colon = f"{_IMAGE_PREFIX}:"
    max_id_len = _DOCKER_TAG_MAX - len(prefix_and_colon)
    tag_id = instance_id[:max_id_len].rstrip("-")
    return f"{prefix_and_colon}{tag_id}"


class TaskSpec(BaseModel):
    """Runtime descriptor for a public-format HALT-Bench task folder.

    Populated by reading instance_id.json from the task directory.
    All file paths are derived from task_dir; files are optional unless noted.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    instance_id: str
    image_tag: str
    task_dir: Path
    support_setup_patch: bool = False
    language: str = "python"

    # ------------------------------------------------------------------
    # Aliases kept for compatibility with agents / evaluation modules
    # that reference task.task_id or task.image_ref.
    # ------------------------------------------------------------------

    @computed_field  # type: ignore[prop-decorator]
    @property
    def task_id(self) -> str:
        return self.instance_id

    @computed_field  # type: ignore[prop-decorator]
    @property
    def image_ref(self) -> str:
        return self.image_tag

    # ------------------------------------------------------------------
    # File path helpers (files may or may not exist — callers must check)
    # ------------------------------------------------------------------

    @property
    def user_request_path(self) -> Path:
        return self.task_dir / "user_request.md"

    @property
    def blocker_registry_path(self) -> Path:
        return self.task_dir / "blocker_registry.json"

    @property
    def setup_patch_path(self) -> Path:
        return self.task_dir / "setup_patch.diff"

    @property
    def setup_script_path(self) -> Path:
        return self.task_dir / "setup_script.sh"

    @property
    def setup_assert_path(self) -> Path:
        return self.task_dir / "setup_assert.sh"

    @property
    def golden_patch_path(self) -> Path:
        return self.task_dir / "golden_patch.diff"

    @property
    def visible_tests_path(self) -> Path:
        return self.task_dir / "visible_tests.json"

    @property
    def run_script_path(self) -> Path:
        return self.task_dir / "run_script.sh"

    @property
    def parser_path(self) -> Path:
        return self.task_dir / "parser.py"


    @classmethod
    def from_task_dir(cls, task_dir: Path) -> "TaskSpec":
        """Load a TaskSpec by reading instance_id.json from task_dir."""
        id_path = task_dir / "instance_id.json"
        data = json.loads(id_path.read_text())
        instance_id: str = data["instance_id"]
        support_setup_patch: bool = bool(data.get("support_setup_patch", False))
        language: str = str(data.get("language", "python")).strip() or "python"
        return cls(
            instance_id=instance_id,
            image_tag=make_image_tag(instance_id),
            task_dir=task_dir.resolve(),
            support_setup_patch=support_setup_patch,
            language=language,
        )
