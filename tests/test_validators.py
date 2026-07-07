"""Tests for the deterministic backbone. Uses SimpleLemmatizer for determinism."""

from islm.config import Thresholds
from islm.validators import validate_story
from islm.vocab.lemmatize import SimpleLemmatizer

LEM = SimpleLemmatizer()
KNOWN = {"the", "cat", "see", "a", "and", "run", "it", "is", "big"}
TARGET = {"clue"}


def _validate(text, known=KNOWN, target=TARGET, thresholds=None):
    return validate_story(text, known, target, LEM, thresholds or Thresholds())


def test_compliant_story_passes_all_checks():
    story = "The cat see a clue. The cat run and run. It is a big clue. The clue is big."
    r = _validate(story)
    assert r.coverage.oov_rate == 0.0
    assert r.coverage.coverage == 1.0
    assert r.one_new_word.passed
    assert r.one_new_word.max_new_words == 1
    assert r.recurrence.counts["clue"] == 3
    assert r.recurrence.passed
    assert r.hard_pass


def test_recurring_target_is_not_counted_as_new_again():
    story = "The cat see a clue. It is a big clue. The clue is big."
    r = _validate(story)
    # "clue" is new only in sentence 0, then recurs without violating the budget.
    assert r.one_new_word.per_sentence[0].new_words == ["clue"]
    assert r.one_new_word.per_sentence[1].new_words == []
    assert r.one_new_word.max_new_words == 1


def test_two_new_words_in_one_sentence_fails():
    # "box" is neither known nor target -> OOV and a second new word in sentence 0.
    r = _validate("The cat see a clue box.")
    assert r.one_new_word.passed is False
    assert r.one_new_word.violations == [0]
    assert set(r.one_new_word.per_sentence[0].new_words) == {"clue", "box"}
    assert r.coverage.oov_words == ["box"]
    assert r.coverage.oov_rate > 0
    assert r.hard_pass is False


def test_recurrence_below_threshold_fails():
    r = _validate("The cat see a clue. The cat run.")  # clue appears once
    assert r.recurrence.counts["clue"] == 1
    assert r.recurrence.below == ["clue"]
    assert r.recurrence.passed is False


def test_proper_nouns_allowed_by_default():
    story = "The cat see Milo. The cat run to Milo."
    r = validate_story(story, {"the", "cat", "see", "run", "to"}, set(), LEM)
    assert r.coverage.proper_nouns == 2
    assert r.coverage.oov == 0
    # Names are not vocabulary to learn, so they never count as new words.
    assert r.one_new_word.max_new_words == 0


def test_failures_report_lists_reasons():
    r = _validate("The cat see a clue box.")
    joined = " ".join(r.failures())
    assert "out-of-vocabulary" in joined
    assert "new word" in joined
