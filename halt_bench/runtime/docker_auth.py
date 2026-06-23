from __future__ import annotations

import os


def get_dockerhub_env_vars() -> dict[str, str]:
    """Return DockerHub credentials from commonly used environment keys."""
    username = (
        os.getenv("DOCKERHUB_USERNAME")
        or os.getenv("DOCKER_USERNAME")
        or os.getenv("DOCKER_HUB_USERNAME")
        or ""
    )
    token = (
        os.getenv("DOCKERHUB_TOKEN")
        or os.getenv("DOCKER_TOKEN")
        or os.getenv("DOCKER_HUB_TOKEN")
        or ""
    )
    return {"DOCKERHUB_USERNAME": username, "DOCKERHUB_TOKEN": token}
