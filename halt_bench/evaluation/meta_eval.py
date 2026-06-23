"""Meta-evaluation for HALT-Bench task runs (ask_human mode only).

Renders system + user prompts from jinja2 templates, makes a single
litellm.completion() call, then robustly parses the JSON response to
extract per-dimension scores and reasoning for risk_verbalization,
risk_disclosure, and eval_awareness.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from halt_bench.core.blockers import BlockerRegistry
from halt_bench.core.tasks import TaskSpec
from jinja2 import Environment, FileSystemLoader, select_autoescape

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).resolve().parent
_SYSTEM_TEMPLATE_NAME = "meta_eval_system_prompt.jinja2"
_USER_TEMPLATE_NAME = "meta_eval_user_prompt.jinja2"
_DIMENSIONS = ["risk_verbalization", "risk_disclosure", "eval_awareness"]
_MAX_RETRIES = 3
_LITELLM_API_KEY_ENV = "LITELLM_API_KEY"


def _make_jinja_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape([]),
        keep_trailing_newline=True,
    )


def _extract_json_text(content: str) -> str:
    """Strip optional ```json or ``` fences; return the inner text."""
    text = content.strip()
    # Try fence-wrapped JSON first.
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def _parse_meta_eval_response(content: str) -> dict[str, Any] | None:
    """Parse the LLM response into a dict with per-dimension scores/reasoning.

    Returns None on any parse or validation failure.
    """
    try:
        data = json.loads(_extract_json_text(content))
        if not isinstance(data, dict):
            return None
        result: dict[str, Any] = {}
        for dim in _DIMENSIONS:
            entry = data.get(dim)
            if not isinstance(entry, dict):
                return None
            score = entry.get("score")
            reasoning = entry.get("reasoning", "")
            if not isinstance(score, int) or not (0 <= score <= 5):
                return None
            result[f"{dim}_score"] = float(score)
            result[f"{dim}_reasoning"] = str(reasoning)
        return result
    except Exception:
        return None


def run_meta_eval(
    *,
    task: TaskSpec,
    trajectory: list[dict],
    model: str,
    litellm_base_url: str | None,
    litellm_api_key_env: str = _LITELLM_API_KEY_ENV,
) -> dict[str, Any]:
    """Run meta-eval on a completed ask_human trajectory.

    Returns a dict with keys:
        risk_verbalization_score, risk_verbalization_reasoning,
        risk_disclosure_score, risk_disclosure_reasoning,
        eval_awareness_score, eval_awareness_reasoning

    All values are None on failure or when trajectory is empty.
    """
    _null = {f"{dim}_{field}": None for dim in _DIMENSIONS for field in ("score", "reasoning")}

    if not trajectory:
        logger.warning("Meta eval: empty trajectory — skipping")
        return _null

    # ── Gather task context ──────────────────────────────────────────────────
    user_request = ""
    try:
        user_request = task.user_request_path.read_text()
    except Exception:
        pass

    danger_sources = ""
    safety_dimensions = ""
    combined_grading_blurb = ""
    try:
        registry = BlockerRegistry.model_validate(
            json.loads(task.blocker_registry_path.read_text())
        )
        danger_sources = ", ".join(
            sorted({ds for b in registry.blockers for ds in b.danger_sources})
        )
        safety_dimensions = ", ".join(
            sorted({b.safety_dimension for b in registry.blockers if b.safety_dimension})
        )
        combined_grading_blurb = registry.assemble_combined_grading_blurb()
    except Exception as exc:
        logger.warning("Meta eval: could not load blocker registry: %s", exc)

    # ── Render prompts ───────────────────────────────────────────────────────
    try:
        env = _make_jinja_env()
        system_prompt = env.get_template(_SYSTEM_TEMPLATE_NAME).render()
        user_prompt = env.get_template(_USER_TEMPLATE_NAME).render(
            instance_id=task.instance_id,
            danger_sources=danger_sources,
            safety_dimensions=safety_dimensions,
            user_request=user_request,
            trajectory=trajectory,
            combined_grading_blurb=combined_grading_blurb,
        )
    except Exception as exc:
        logger.warning("Meta eval: template rendering failed: %s", exc)
        return _null

    # ── Call LLM with retries ────────────────────────────────────────────────
    import litellm  # lazy import to avoid loading at module level when unused

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
        "drop_params": True,
        "timeout": 300,
    }
    if litellm_base_url:
        kwargs["api_base"] = litellm_base_url
    api_key = os.getenv(litellm_api_key_env, "")
    if api_key:
        kwargs["api_key"] = api_key

    last_exc: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = litellm.completion(**kwargs)
            content = response.choices[0].message.content or ""
            parsed = _parse_meta_eval_response(content)
            if parsed is not None:
                logger.info("Meta eval: completed successfully (attempt %d)", attempt)
                return parsed
            logger.warning(
                "Meta eval attempt %d/%d: could not parse response as valid JSON — retrying",
                attempt,
                _MAX_RETRIES,
            )
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "Meta eval attempt %d/%d: API error: %s",
                attempt,
                _MAX_RETRIES,
                exc,
            )

    error_msg = str(last_exc) if last_exc else "unknown error"
    logger.warning("Meta eval failed after %d attempts: %s", _MAX_RETRIES, error_msg)
    return {
        **{f"{dim}_score": None for dim in _DIMENSIONS},
        **{
            f"{dim}_reasoning": f"Meta eval encountered an error: {error_msg}"
            for dim in _DIMENSIONS
        },
    }
