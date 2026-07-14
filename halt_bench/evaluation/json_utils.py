from __future__ import annotations

import json
import re
from typing import Any


def parse_json_object_response(content: Any) -> dict | None:
    """Parse a JSON object from an LLM response."""
    text = str(content).strip()
    if not text:
        return None

    candidates: list[str] = [text]
    for match in re.finditer(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE):
        fenced = match.group(1).strip()
        if fenced:
            candidates.insert(0, fenced)

    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        for idx, char in enumerate(candidate):
            if char != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(candidate[idx:])
            except Exception:
                continue
            if isinstance(parsed, dict):
                return parsed

    return None
