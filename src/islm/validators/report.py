"""Aggregate validator: turns a story into a pass/fail ValidationReport."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ..config import DEFAULT_THRESHOLDS, Thresholds
from ..vocab.lemmatize import Lemmatizer, get_lemmatizer
from ..vocab.tokenize import split_sentences
from .coverage import CoverageResult, coverage
from .one_new_word import OneNewWordResult, one_new_word
from .recurrence import RecurrenceResult, recurrence


@dataclass
class ValidationReport:
    coverage: CoverageResult
    one_new_word: OneNewWordResult
    recurrence: RecurrenceResult
    thresholds: Thresholds

    @property
    def hard_pass(self) -> bool:
        """True iff every deterministic check meets its threshold."""
        c = self.coverage
        return (
            c.oov_rate <= self.thresholds.max_oov_rate
            and c.coverage >= self.thresholds.min_coverage
            and self.one_new_word.passed
            and self.recurrence.passed
        )

    def failures(self) -> list[str]:
        """Human-readable failure reasons (also fed back into the rewrite loop)."""
        out: list[str] = []
        t, c = self.thresholds, self.coverage
        if c.oov_rate > t.max_oov_rate:
            out.append(
                f"OOV rate {c.oov_rate:.3f} > {t.max_oov_rate}; "
                f"out-of-vocabulary words: {c.oov_words}"
            )
        if c.coverage < t.min_coverage:
            out.append(f"coverage {c.coverage:.3f} < {t.min_coverage}")
        if not self.one_new_word.passed:
            details = {
                s.index: s.new_words
                for s in self.one_new_word.per_sentence
                if s.index in self.one_new_word.violations
            }
            out.append(
                f"sentences with more than {t.max_new_words_per_sentence} new word(s): {details}"
            )
        if not self.recurrence.passed:
            below = {w: self.recurrence.counts[w] for w in self.recurrence.below}
            out.append(f"target words below {t.min_recurrence} occurrences: {below}")
        return out

    def to_dict(self) -> dict:
        return {
            "hard_pass": self.hard_pass,
            "coverage": asdict(self.coverage),
            "one_new_word": asdict(self.one_new_word),
            "recurrence": asdict(self.recurrence),
            "failures": self.failures(),
        }


def validate_story(
    text: str,
    known: set[str],
    target: set[str],
    lemmatizer: Lemmatizer | None = None,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> ValidationReport:
    """Run all deterministic checks on a story. This is the pipeline's backbone."""
    lemmatizer = lemmatizer or get_lemmatizer()
    known = {w.lower() for w in known}
    target = {w.lower() for w in target}
    sentences = [lemmatizer.analyze(s) for s in split_sentences(text)]
    return ValidationReport(
        coverage=coverage(sentences, known, target, thresholds.allow_proper_nouns),
        one_new_word=one_new_word(sentences, known, thresholds.max_new_words_per_sentence),
        recurrence=recurrence(sentences, target, thresholds.min_recurrence),
        thresholds=thresholds,
    )
