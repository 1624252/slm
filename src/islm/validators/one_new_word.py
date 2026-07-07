"""At most one new word per sentence (the core i+1 pacing rule).

A word is "new" the first time it appears in the story and is not in the known set.
Later repetitions are not new, so intentional recurrence never trips this check.
Proper nouns are treated as names, not vocabulary to learn.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from ..vocab.lemmatize import LemmaToken


@dataclass
class SentenceNewWords:
    index: int
    new_words: list[str] = field(default_factory=list)


@dataclass
class OneNewWordResult:
    per_sentence: list[SentenceNewWords]
    max_new_words: int
    violations: list[int]  # sentence indices exceeding the budget
    passed: bool


def one_new_word(
    sentences: Iterable[Sequence[LemmaToken]],
    known: set[str],
    max_per_sentence: int = 1,
) -> OneNewWordResult:
    seen: set[str] = set()  # non-known lemmas already introduced earlier
    per_sentence: list[SentenceNewWords] = []
    violations: list[int] = []
    max_new = 0

    for i, tokens in enumerate(sentences):
        new: list[str] = []
        new_set: set[str] = set()
        for tok in tokens:
            if not tok.is_word or tok.is_proper:
                continue
            if tok.lemma in known or tok.surface.lower() in known:
                continue
            if tok.lemma in seen or tok.lemma in new_set:
                continue
            new.append(tok.lemma)
            new_set.add(tok.lemma)
        seen |= new_set
        per_sentence.append(SentenceNewWords(i, new))
        max_new = max(max_new, len(new))
        if len(new) > max_per_sentence:
            violations.append(i)

    return OneNewWordResult(per_sentence, max_new, violations, not violations)
