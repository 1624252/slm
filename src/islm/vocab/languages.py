"""Language registry.

The pipeline is language-agnostic; a `Language` entry just wires up the right analyzer,
level scheme, and known/target tiers. English, Chinese, and Japanese are the shipped set, and
any other code falls back to a generic (frequency-based, whitespace-tokenized) profile.
"""

from __future__ import annotations

from dataclasses import dataclass

# Narrow-reading themes (Krashen 2004). Kept as English prompt hints; the teacher writes the
# story in the target language. Localize per language later if desired.
_THEMES = (
    "a cat detective",
    "a lost key",
    "a funny dog",
    "a quiet garden",
    "a rainy day",
    "a new friend",
    "a night in the old house",
    "a bird and a tree",
    "a mouse and the cheese",
    "a walk in the park",
)


@dataclass(frozen=True)
class Language:
    code: str  # ISO code, e.g. "en"
    name: str
    script: str  # latin | han | japanese | other
    analyzer: str  # which analyzer to use: en | zh | ja | generic
    level_scheme: str  # CEFR | HSK | JLPT | frequency
    baseline_tiers: tuple[str, ...]  # levels treated as already-known
    advanced_tiers: tuple[str, ...]  # levels treated as to-learn
    themes: tuple[str, ...] = _THEMES
    # Fallback sizes when no curated list exists: baseline = top N by frequency,
    # advanced = words ranked (freq_baseline_n, freq_advanced_n].
    freq_baseline_n: int = 1500
    freq_advanced_n: int = 6000


LANGUAGES: dict[str, Language] = {
    "en": Language("en", "English", "latin", "en", "CEFR", ("A1", "A2"), ("B1", "B2", "C1")),
    "zh": Language(
        "zh",
        "Chinese (Mandarin)",
        "han",
        "zh",
        "HSK",
        ("HSK1", "HSK2", "HSK3"),
        ("HSK4", "HSK5", "HSK6"),
    ),
    "ja": Language(
        "ja",
        "Japanese",
        "japanese",
        "ja",
        "JLPT",
        ("N5", "N4"),
        ("N3", "N2", "N1"),
    ),
}

SUPPORTED_LANGUAGES = tuple(LANGUAGES)


def get_language(code: str) -> Language:
    """Return a registered language, or a generic profile for anything else."""
    code = code.lower()
    if code in LANGUAGES:
        return LANGUAGES[code]
    return Language(code, code, "other", "generic", "frequency", (), ())
