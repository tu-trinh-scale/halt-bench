from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class BlockerEntry(BaseModel):
    id: str
    description: str = ""
    resolution: str
    grading_blurb: str = ""
    danger_sources: list[str] = Field(default_factory=list)
    safety_dimension: str = ""
    trigger_statements: list[str] = Field(default_factory=list)


class BlockerRegistry(BaseModel):
    blockers: list[BlockerEntry] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _accept_flat_list(cls, v: object) -> object:
        """Canonical format is a flat JSON array. Reject the legacy dict wrapper."""
        if isinstance(v, list):
            return {"blockers": v}
        if isinstance(v, dict) and "blockers" in v:
            raise ValueError(
                "blocker_registry.json must be a flat JSON array, not a dict with a "
                "'blockers' key. Run reformat_tasks.py to convert it."
            )
        return v

    def to_dict(self) -> dict[str, BlockerEntry]:
        return {blocker.id: blocker for blocker in self.blockers}

    def to_list(self) -> list[dict]:
        """Serialize to a flat list of blocker dicts."""
        return [b.model_dump() for b in self.blockers]

    def assemble_combined_grading_blurb(self) -> str:
        """Concatenate per-blocker grading blurbs into one numbered-section string."""
        parts = []
        for i, blocker in enumerate(self.blockers, 1):
            if blocker.grading_blurb.strip():
                parts.append(f"## Blocker {i}: {blocker.id}\n{blocker.grading_blurb.strip()}")
        return "\n\n".join(parts)
