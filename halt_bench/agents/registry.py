from __future__ import annotations

from pathlib import Path

from halt_bench.agents.base import HaltAgent
from halt_bench.agents.opencode.types import OpenCodeConfig
from halt_bench.agents.opencode_agent import OpenCodeAgent


def build_agent(
    agent_name: str,
    *,
    simulate: bool = False,
    model: str = "gpt-5.5",
    max_steps: int = 200,
    litellm_base_url: str | None = None,
    use_local_drop_params_proxy: bool = True,
    llm_total_tries: int = 3,
    llm_timeout_seconds: int = 300,
    run_timeout_seconds: int | None = 1800,
    with_custom_tool: bool = False,
    with_ask_guidance: bool = False,
    ask_guidance_path: Path | None = None,
    solver_temperature: float | None = None,
    solver_max_tokens: int | None = None,
) -> HaltAgent:
    normalized = agent_name.strip().lower()
    if normalized == "opencode":
        return OpenCodeAgent(
            OpenCodeConfig(
                model=model,
                max_steps=max_steps,
                simulate=simulate,
                litellm_base_url=litellm_base_url,
                use_local_litellm_drop_params_proxy=use_local_drop_params_proxy,
                llm_total_tries=llm_total_tries,
                llm_timeout_seconds=llm_timeout_seconds,
                run_timeout_seconds=run_timeout_seconds,
                with_custom_tool=with_custom_tool,
                with_ask_guidance=with_ask_guidance,
                ask_guidance_path=ask_guidance_path if with_ask_guidance else None,
                solver_temperature=solver_temperature,
                solver_max_tokens=solver_max_tokens,
            )
        )
    raise ValueError(f"Unsupported agent: {agent_name}")
