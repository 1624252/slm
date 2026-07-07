"""Lexical coverage / out-of-vocabulary rate (Nation 2006: 98% coverage target)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ..vocab.lemmatize import LemmaToken


@dataclass
class CoverageResult:
    total_words: int
    known: int
    target: int
    proper_nouns: int
    oov: int
    oov_words: list[str]  # distinct, in order of first appearance
    coverage: float  # fraction of words the learner can understand
    oov_rate: float  # fraction outside known + target (+ proper nouns)


def _is_cjk(ch: str) -> bool:
    o = ord(ch)
    return 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0x3040 <= o <= 0x30FF


def is_known(vocab: set[str], token: LemmaToken) -> bool:
    """Is a token covered by a vocabulary?

    Accepts the lemma or the raw lowercase surface (absorbs imperfect lemmatization). For a
    multi-character CJK token, also accepts it when every character is individually known — this
    bridges the gap between segmenter granularity (jieba/MeCab merge or split differently) and the
    HSK/JLPT word-list granularity, so natural CJK text isn't falsely flagged OOV.
    """
    if token.lemma in vocab or token.surface.lower() in vocab:
        return True
    surface = token.surface
    if len(surface) >= 2 and all(_is_cjk(c) for c in surface):
        return all(c in vocab for c in surface)
    return False


def coverage(
    sentences: Iterable[Sequence[LemmaToken]],
    known: set[str],
    target: set[str],
    allow_proper_nouns: bool = True,
) -> CoverageResult:
    total = known_c = target_c = proper_c = oov_c = 0
    oov_words: list[str] = []
    seen_oov: set[str] = set()

    for tokens in sentences:
        for tok in tokens:
            if not tok.is_word:
                continue
            total += 1
            if is_known(target, tok):
                target_c += 1
            elif is_known(known, tok):
                known_c += 1
            elif allow_proper_nouns and tok.is_proper:
                proper_c += 1
            else:
                oov_c += 1
                if tok.lemma not in seen_oov:
                    seen_oov.add(tok.lemma)
                    oov_words.append(tok.lemma)

    understood = known_c + target_c + proper_c
    return CoverageResult(
        total_words=total,
        known=known_c,
        target=target_c,
        proper_nouns=proper_c,
        oov=oov_c,
        oov_words=oov_words,
        coverage=understood / total if total else 1.0,
        oov_rate=oov_c / total if total else 0.0,
    )
