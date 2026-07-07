"""Known/target vocabulary sources.

Per language we read curated tier files at ``data/vocab/<lang>/{baseline,advanced}.csv``
(columns: ``word,tier,source``). If a file is missing we fall back to frequency bands via
``wordfreq`` (which supports ~any language), so the system degrades gracefully everywhere.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from ..config import DATA_DIR
from .languages import get_language

VOCAB_DIR = DATA_DIR / "vocab"
_WORD_COLS = ("word", "headword", "hanzi", "surface")
_TIER_COLS = ("tier", "cefr", "level", "hsk", "jlpt")


@dataclass
class Vocabulary:
    """A set of lowercase lemmas, optionally tagged with tier labels."""

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
    def from_words(cls, words, tier: str | None = None) -> Vocabulary:
        lemmas = {w.strip().lower() for w in words if w and w.strip()}
        levels = {w: tier for w in lemmas} if tier else {}
        return cls(lemmas, levels)

    @classmethod
    def from_csv(cls, path: Path) -> Vocabulary:
        """Load a CSV whose columns include a word column and (optionally) a tier column."""
        lemmas: set[str] = set()
        levels: dict[str, str] = {}
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            word_col = _pick(reader.fieldnames, _WORD_COLS)
            tier_col = _pick(reader.fieldnames, _TIER_COLS)
            if word_col is None:
                raise ValueError(f"{path}: no word column among {_WORD_COLS}")
            for row in reader:
                word = (row.get(word_col) or "").strip().lower()
                if not word:
                    continue
                lemmas.add(word)
                if tier_col:
                    tier = (row.get(tier_col) or "").strip()
                    if tier:
                        levels[word] = tier
        return cls(lemmas, levels)

    @classmethod
    def from_frequency(cls, language: str = "en", top_n: int = 2000) -> Vocabulary:
        """Known set = the `top_n` most frequent words (Nation's frequency bands)."""
        from wordfreq import top_n_list  # lazy: keep wordfreq optional

        return cls.from_words(top_n_list(language, top_n))


def _pick(fieldnames, candidates) -> str | None:
    if not fieldnames:
        return None
    lowered = {name.lower(): name for name in fieldnames}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    return None


def _load_tier(language: str, tier: str) -> Vocabulary:
    """Union of the downloaded full list and the committed curated sample (either may be absent)."""
    vocab = Vocabulary()
    for name in (f"{tier}.full.csv", f"{tier}.csv"):
        path = VOCAB_DIR / language / name
        if path.exists():
            vocab = vocab | Vocabulary.from_csv(path)
    return vocab


def load_baseline(language: str) -> Vocabulary:
    """Common 'already known' vocabulary (curated + full files, else frequency fallback)."""
    vocab = _load_tier(language, "baseline")
    if vocab.lemmas:
        return vocab
    return Vocabulary.from_frequency(language, top_n=get_language(language).freq_baseline_n)


def load_advanced(language: str) -> Vocabulary:
    """Graded 'to-learn' vocabulary (curated + full files, else frequency-band fallback).

    Anything already in the baseline is removed, so the tiers are always disjoint (a to-learn
    word is never also a known word).
    """
    vocab = _load_tier(language, "advanced")
    if not vocab.lemmas:
        lang = get_language(language)
        from wordfreq import top_n_list  # lazy

        band = top_n_list(language, lang.freq_advanced_n)[lang.freq_baseline_n :]
        vocab = Vocabulary.from_words(band)
    return Vocabulary(vocab.lemmas - load_baseline(language).lemmas)
