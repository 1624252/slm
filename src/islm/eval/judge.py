"""LLM-as-judge scoring against the Behavior Spec rubric (PRD 14.3)."""

from __future__ import annotations

import json
import re

from ..datagen.scenarios import Scenario
from ..llm.client import LLMClient
from ..llm.prompts import JUDGE_DIMENSIONS, SPEC_DIMENSIONS, judge_prompt

# Re-exported for the harness/report. SPEC_DIMENSIONS drive the win condition (spec Appendix A).
DIMENSIONS = JUDGE_DIMENSIONS

__all__ = ["DIMENSIONS", "SPEC_DIMENSIONS", "judge_story"]


def _extract_json(raw: str) -> dict:
    # Tolerate code fences or surrounding prose; grab the first {...} block.
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def judge_story(scenario: Scenario, story: str, client: LLMClient) -> dict:
    """Return integer 0-2 scores for each rubric dimension plus a rationale."""
    system, user = judge_prompt(scenario, story)
    parsed = _extract_json(client.complete(system, user, temperature=0.0, max_tokens=400))

    scores: dict = {}
    for dim in DIMENSIONS:
        try:
            scores[dim] = max(0, min(2, int(parsed.get(dim, 0))))
        except (TypeError, ValueError):
            scores[dim] = 0
    scores["rationale"] = str(parsed.get("rationale", ""))[:500]
    return scores
