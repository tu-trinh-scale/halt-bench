from __future__ import annotations

from pydantic import BaseModel, Field


class LlmSafetyGradingResult(BaseModel):
    agent_passes: bool
    reasoning: str
    steps_taken: int | None = None
    num_blockers: int | None = None
    num_blockers_failed: int | None = None
    blocker_ids_failed: list[str] | None = None


class TestCorrectnessResult(BaseModel):
    passed: bool
    detail: str
    # True when visible_tests=[] (no tests to fail → always passing).
    visible_passed: bool = True
    # True when tests could not be collected/evaluated at all (e.g. ImportError
    # at collection time).  Signals that the run outcome is indeterminate rather
    # than a clean agent failure; the controller treats this as inconclusive.
    tc_inconclusive: bool = False
    # Full per-test output text (PASS/FAIL per test) for gist upload. None when
    # the parser produced no structured output (exit-code fallback path).
    test_output_text: str | None = None


class EvaluationResult(BaseModel):
    # ── Safety grading (all modes) ────────────────────────────────────────────
    # Fraction of blockers the agent triggered (FAIL verdicts / total blockers).
    harm_execution_rate: float | None = None
    safety_grading_reasoning: str = ""
    safety_grading: LlmSafetyGradingResult

    # ── Ask-human interaction metrics (ask_human mode only; None for full_info) ─
    harm_discovery_rate: float | None = None
    question_efficiency_rate: float | None = None
    num_questions: int | None = None
    blockers_resolved: list[str] = Field(default_factory=list)

    # ── Meta-eval scores (ask_human mode only; None for full_info) ────────────
    risk_verbalization_score: float | None = None
    risk_verbalization_reasoning: str | None = None
    risk_disclosure_score: float | None = None
    risk_disclosure_reasoning: str | None = None
    eval_awareness_score: float | None = None
    eval_awareness_reasoning: str | None = None

    # ── Test correctness ──────────────────────────────────────────────────────
    test_correctness: TestCorrectnessResult | None = None
