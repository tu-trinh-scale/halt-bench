from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def load_dotenv_file(path: Path) -> None:
    load_dotenv(dotenv_path=path, override=False)


def load_project_env(project_root: Path) -> None:
    load_dotenv_file(project_root / ".env")
