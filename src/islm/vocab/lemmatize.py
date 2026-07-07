"""Lemmatizers.

`SimpleLemmatizer` is dependency-free (used by tests and offline smoke runs).
`SpacyLemmatizer` is recommended for real data: it is POS-aware and handles irregular
forms and sentence-initial proper nouns that the rule-based fallback cannot.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from .tokenize import word_surfaces


@dataclass(frozen=True)
class LemmaToken:
    surface: str
    lemma: str
    is_word: bool = True
    is_proper: bool = False


class Lemmatizer(Protocol):
    def analyze(self, sentence: str) -> list[LemmaToken]:
        """Tokenize a single sentence into word tokens with lowercase lemmas."""
        ...


# Common irregular forms likely to appear in simple narratives.
_IRREGULAR: dict[str, str] = {
    "am": "be",
    "is": "be",
    "are": "be",
    "was": "be",
    "were": "be",
    "been": "be",
    "being": "be",
    "has": "have",
    "had": "have",
    "having": "have",
    "does": "do",
    "did": "do",
    "done": "do",
    "doing": "do",
    "goes": "go",
    "went": "go",
    "gone": "go",
    "going": "go",
    "ate": "eat",
    "eaten": "eat",
    "eating": "eat",
    "eats": "eat",
    "ran": "run",
    "running": "run",
    "runs": "run",
    "saw": "see",
    "seen": "see",
    "sees": "see",
    "seeing": "see",
    "made": "make",
    "makes": "make",
    "making": "make",
    "said": "say",
    "says": "say",
    "saying": "say",
    "found": "find",
    "finds": "find",
    "finding": "find",
    "took": "take",
    "taken": "take",
    "takes": "take",
    "taking": "take",
    "came": "come",
    "comes": "come",
    "coming": "come",
    "got": "get",
    "gotten": "get",
    "gets": "get",
    "getting": "get",
    "gave": "give",
    "given": "give",
    "gives": "give",
    "giving": "give",
    "knew": "know",
    "known": "know",
    "knows": "know",
    "knowing": "know",
    "thought": "think",
    "thinks": "think",
    "thinking": "think",
    "told": "tell",
    "tells": "tell",
    "telling": "tell",
    "felt": "feel",
    "feels": "feel",
    "feeling": "feel",
    "left": "leave",
    "leaves": "leave",
    "leaving": "leave",
    "sat": "sit",
    "sits": "sit",
    "sitting": "sit",
    "stood": "stand",
    "stands": "stand",
    "standing": "stand",
    "hid": "hide",
    "hidden": "hide",
    "hides": "hide",
    "hiding": "hide",
    "caught": "catch",
    "catches": "catch",
    "catching": "catch",
    "slept": "sleep",
    "sleeps": "sleep",
    "sleeping": "sleep",
    "children": "child",
    "mice": "mouse",
    "feet": "foot",
    "teeth": "tooth",
    "men": "man",
    "women": "woman",
    "people": "person",
    "geese": "goose",
    "better": "good",
    "best": "good",
    "worse": "bad",
    "worst": "bad",
    "bigger": "big",
    "biggest": "big",
    "happier": "happy",
    "happiest": "happy",
}

_VOWELS = frozenset("aeiou")


def _simple_base(w: str) -> str:
    """Best-effort English lemma via small irregular table + suffix rules."""
    if w in _IRREGULAR:
        return _IRREGULAR[w]
    if w.endswith(("ies", "ied")) and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("ing") and len(w) > 5:
        base = w[:-3]
        if len(base) >= 2 and base[-1] == base[-2] and base[-1] not in _VOWELS:
            base = base[:-1]  # running -> run
        return base
    if w.endswith("ed") and len(w) > 3:
        base = w[:-2]
        if len(base) >= 2 and base[-1] == base[-2] and base[-1] not in _VOWELS:
            base = base[:-1]  # stopped -> stop
        return base
    if w.endswith(("ches", "shes", "xes", "zes", "ses", "oes")) and len(w) > 4:
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    if w.endswith("est") and len(w) > 5:
        return w[:-3]
    if w.endswith("er") and len(w) > 4:
        return w[:-2]
    return w


class SimpleLemmatizer:
    """Rule-based English lemmatizer. Imperfect (e.g. silent-e verbs) but deterministic."""

    def analyze(self, sentence: str) -> list[LemmaToken]:
        out: list[LemmaToken] = []
        for i, surface in enumerate(word_surfaces(sentence)):
            low = surface.lower()
            # Heuristic proper noun: capitalized and not sentence-initial.
            is_proper = surface[:1].isupper() and i > 0
            lemma = low if is_proper else _simple_base(low)
            out.append(LemmaToken(surface, lemma, True, is_proper))
        return out


class SpacyLemmatizer:
    """POS-aware lemmatizer backed by spaCy (recommended for real data)."""

    def __init__(self, model: str = "en_core_web_sm"):
        import spacy  # lazy: keep spaCy optional

        try:
            # Parser/NER not needed; we analyze one sentence at a time.
            self._nlp = spacy.load(model, disable=["parser", "ner"])
        except OSError as exc:
            raise RuntimeError(
                f"spaCy model '{model}' not installed. Run: python -m spacy download {model}"
            ) from exc

    def analyze(self, sentence: str) -> list[LemmaToken]:
        return [
            LemmaToken(t.text, t.lemma_.lower(), True, t.pos_ == "PROPN")
            for t in self._nlp(sentence)
            if t.is_alpha
        ]


def get_lemmatizer(prefer_spacy: bool = True, model: str = "en_core_web_sm") -> Lemmatizer:
    """Return spaCy lemmatizer if available, else the rule-based fallback (English)."""
    if prefer_spacy:
        try:
            return SpacyLemmatizer(model)
        except Exception:
            pass
    return SimpleLemmatizer()


# --- Chinese / Japanese / generic analyzers -------------------------------------------------


def _is_cjk(ch: str) -> bool:
    o = ord(ch)
    return (
        0x4E00 <= o <= 0x9FFF  # CJK unified ideographs
        or 0x3400 <= o <= 0x4DBF  # extension A
        or 0x3040 <= o <= 0x30FF  # hiragana + katakana
    )


class GenericAnalyzer:
    """Unicode word tokenizer for space-delimited scripts; lemma = lowercase surface."""

    _WORD = re.compile(r"\w+", re.UNICODE)

    def analyze(self, sentence: str) -> list[LemmaToken]:
        return [
            LemmaToken(s, s.lower(), True, s[:1].isupper() and i > 0)
            for i, s in enumerate(self._WORD.findall(sentence))
        ]


class CjkCharAnalyzer:
    """Offline fallback for Chinese/Japanese: one token per CJK character.

    Crude (misses multi-character words) but needs no external segmenter. Prefer
    ChineseAnalyzer / JapaneseAnalyzer for real data.
    """

    def analyze(self, sentence: str) -> list[LemmaToken]:
        return [LemmaToken(ch, ch, True, False) for ch in sentence if _is_cjk(ch)]


class ChineseAnalyzer:
    """Word segmentation via jieba, with POS-based proper-noun detection."""

    def __init__(self):
        import logging
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import jieba
            import jieba.posseg as posseg
        jieba.setLogLevel(logging.WARNING)
        self._posseg = posseg

    def analyze(self, sentence: str) -> list[LemmaToken]:
        out: list[LemmaToken] = []
        for tok in self._posseg.cut(sentence):
            word = tok.word.strip()
            if not word or not (any(_is_cjk(c) for c in word) or word.isalnum()):
                continue  # skip punctuation / whitespace
            is_proper = tok.flag.startswith(("nr", "ns", "nt", "nz"))
            out.append(LemmaToken(word, word.lower(), True, is_proper))
        return out


class JapaneseAnalyzer:
    """Morphological analysis via fugashi (MeCab + UniDic); lemma = dictionary form.

    Particles (助詞) and auxiliaries (助動詞) are skipped: they are grammar, not vocabulary,
    and are not present in JLPT word lists, so counting them would create false OOV hits.
    """

    _SKIP_POS = {"補助記号", "空白", "記号", "助詞", "助動詞"}

    def __init__(self):
        import fugashi

        self._tagger = fugashi.Tagger()

    def analyze(self, sentence: str) -> list[LemmaToken]:
        out: list[LemmaToken] = []
        for word in self._tagger(sentence):
            surface = word.surface.strip()
            if not surface:
                continue
            feat = word.feature
            pos1 = getattr(feat, "pos1", "") or ""
            if pos1 in self._SKIP_POS:
                continue
            lemma = (getattr(feat, "lemma", None) or surface).split("-")[0]
            is_proper = pos1 == "名詞" and getattr(feat, "pos2", "") == "固有名詞"
            out.append(LemmaToken(surface, lemma.lower(), True, is_proper))
        return out


def get_analyzer(language: str = "en", prefer_external: bool = True) -> Lemmatizer:
    """Return the best available analyzer for a language, degrading gracefully."""
    lang = language.lower()
    if lang == "en":
        return get_lemmatizer(prefer_spacy=prefer_external)
    if lang == "zh":
        if prefer_external:
            try:
                return ChineseAnalyzer()
            except Exception:
                pass
        return CjkCharAnalyzer()
    if lang == "ja":
        if prefer_external:
            try:
                return JapaneseAnalyzer()
            except Exception:
                pass
        return CjkCharAnalyzer()
    return GenericAnalyzer()
