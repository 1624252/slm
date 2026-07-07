"""Dependency-free sentence and word tokenization for simple learner stories."""

from __future__ import annotations

import re

# Split after ., !, ? followed by whitespace, or on blank lines.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

# A word is letters, optionally joined by an internal apostrophe or hyphen.
_WORD = re.compile(r"[A-Za-z]+(?:['\u2019-][A-Za-z]+)*")


def split_sentences(text: str) -> list[str]:
    """Split text into non-empty, stripped sentences."""
    return [p.strip() for p in _SENT_SPLIT.split(text.strip()) if p.strip()]


def word_surfaces(text: str) -> list[str]:
    """Return word-like tokens (drops punctuation, numbers, symbols)."""
    return _WORD.findall(text)
