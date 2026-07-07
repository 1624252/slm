"""Target-word recurrence for spaced repetition (>=3 per story; ~8 across a series)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ..vocab.lemmatize import LemmaToken


@dataclass
class RecurrenceResult:
    counts: dict[str, int]
    min_required: int
    below: list[str]  # target words appearing fewer than min_required times
    absent: list[str]  # target words that never appear
    passed: bool


def recurrence(
    sentences: Iterable[Sequence[LemmaToken]],
    target: set[str],
    min_required: int = 3,
) -> RecurrenceResult:
    counts: dict[str, int] = {w: 0 for w in target}

    for tokens in sentences:
        for tok in tokens:
            if not tok.is_word:
                continue
            if tok.lemma in counts:
                counts[tok.lemma] += 1
            elif tok.surface.lower() in counts:
                counts[tok.surface.lower()] += 1

    below = [w for w, c in counts.items() if c < min_required]
    absent = [w for w, c in counts.items() if c == 0]
    return RecurrenceResult(counts, min_required, below, absent, not below)
