"""Known/target vocabulary sources: CSV (CEFR-J style), frequency bands, bundled sample."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from ..config import DATA_DIR

_CEFR_ORDER = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}
SAMPLE_CSV = DATA_DIR / "vocab" / "sample_en.csv"


@dataclass
class Vocabulary:
    """A set of lowercase lemmas, optionally tagged with CEFR levels."""

    lemmas: set[str] = field(default_factory=set)
    levels: dict[str, str] = field(default_factory=dict)

    def __contains__(self, word: str) -> bool:
        return word.lower() in self.lemmas

    def __len__(self) -> int:
        return len(self.lemmas)

    def __or__(self, other: Vocabulary) -> Vocabulary:
        merged = dict(self.levels)
        merged.update(other.levels)
        return Vocabulary(self.lemmas | other.lemmas, merged)

    @classmethod
    def from_words(cls, words) -> Vocabulary:
        return cls({w.strip().lower() for w in words if w and w.strip()})

    @classmethod
    def from_csv(cls, path: Path = SAMPLE_CSV, level_at_most: str | None = None) -> Vocabulary:
        """Load a CSV with columns: headword, pos, cefr. `level_at_most` filters by CEFR band."""
        cap = _CEFR_ORDER.get((level_at_most or "").upper(), 99)
        lemmas: set[str] = set()
        levels: dict[str, str] = {}
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                word = (row.get("headword") or "").strip().lower()
                if not word:
                    continue
                level = (row.get("cefr") or "").strip().upper()
                if level_at_most and _CEFR_ORDER.get(level, 99) > cap:
                    continue
                lemmas.add(word)
                if level:
                    levels[word] = level
        return cls(lemmas, levels)

    @classmethod
    def from_frequency(cls, language: str = "en", top_n: int = 2000) -> Vocabulary:
        """Known set = the `top_n` most frequent words (Nation's frequency bands)."""
        from wordfreq import top_n_list  # lazy: keep wordfreq optional

        return cls.from_words(top_n_list(language, top_n))
