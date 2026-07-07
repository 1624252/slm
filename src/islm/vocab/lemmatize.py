"""Lemmatizers.

`SimpleLemmatizer` is dependency-free (used by tests and offline smoke runs).
`SpacyLemmatizer` is recommended for real data: it is POS-aware and handles irregular
forms and sentence-initial proper nouns that the rule-based fallback cannot.
"""

from __future__ import annotations

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
    """Return spaCy lemmatizer if available, else the rule-based fallback."""
    if prefer_spacy:
        try:
            return SpacyLemmatizer(model)
        except Exception:
            pass
    return SimpleLemmatizer()
