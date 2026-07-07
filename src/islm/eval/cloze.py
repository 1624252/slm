"""Cloze-based inferability proxy: blank a target word and see if it's recoverable from context."""

from __future__ import annotations

import re

from ..llm.client import LLMClient
from ..llm.prompts import cloze_prompt


def _mask(text: str, word: str) -> str:
    # Blank every inflected form of the word (whole-word, case-insensitive).
    return re.sub(rf"\b{re.escape(word)}\w*\b", "____", text, flags=re.IGNORECASE)


def cloze_inferability(story: str, target: set[str], client: LLMClient) -> dict:
    """Fraction of target words a model can recover from context alone."""
    per_word: dict[str, bool] = {}
    for word in target:
        guess = (
            client.complete(*cloze_prompt(_mask(story, word)), temperature=0.0, max_tokens=8)
            .strip()
            .lower()
        )
        per_word[word] = bool(guess) and (guess.startswith(word.lower()) or word.lower() in guess)

    recovered = sum(per_word.values())
    total = len(per_word)
    return {
        "per_word": per_word,
        "recovered": recovered,
        "total": total,
        "rate": recovered / total if total else 1.0,
    }
