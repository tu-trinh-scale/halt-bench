from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class OpenCodeConfig(BaseModel):
    model: str = "gpt-5.5"
    max_steps: int = 200
    llm_total_tries: int = 3
    llm_timeout_seconds: int = 300
    run_timeout_seconds: int | None = 7200
    native_question_poll_interval_ms: int = 200
    with_custom_tool: bool = False
    with_ask_guidance: bool = False
    simulate: bool = False
    litellm_base_url: str | None = None
    use_local_litellm_drop_params_proxy: bool = True
    litellm_api_key_env: str = "LITELLM_API_KEY"
    ask_human_mcp_bridge_script: Path = Path(__file__).resolve().parent / "ask_human_mcp_bridge.mjs"
    litellm_drop_proxy_script: Path = (
        Path(__file__).resolve().parent / "litellm_responses_proxy.mjs"
    )
    sdk_runner_script: Path = Path(__file__).resolve().parent / "run_opencode_sdk.mjs"
    ask_guidance_path: Path | None = None
    # Solver agent temperature. Default is None which is usually 0 for OpenCode for most models.
    solver_temperature: float | None = None
    # Solver max output tokens per LLM response turn. Default is None which is no limit.
    solver_max_tokens: int | None = None


class TrajectoryStep(BaseModel):
    thought: str
    act: str
    obs: str
