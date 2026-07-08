"""Eval coverage per word-list set — one category from each vocabulary we ship.

Every shipped word list must be exercised by the eval harness, so no set can silently rot:
  - en CEFR   : known = A1-A2 baseline, target = B1-C2 advanced
  - en EXAM   : known = A1-A2 baseline, target = GRE/SAT/ACT exam words  (the hard set)
  - zh HSK    : known = HSK1-3 baseline, target = HSK4-9 advanced
  - ja JLPT   : known = N5-N4 baseline, target = N3-N1 advanced

Each runs the real `evaluate()` over MockLLM-generated stories (deterministic, offline). CJK
categories skip cleanly when jieba/fugashi are absent. The English exam category is REQUIRED:
it proves the GRE/SAT/ACT words load, sample as targets, and flow through the eval end to end.
"""

import pytest

from islm.datagen.scenarios import sample_scenarios
from islm.eval import api_generator, evaluate
from islm.vocab.lemmatize import get_analyzer
from islm.vocab.wordlists import VOCAB_DIR, Vocabulary

_EXAM_TIERS = {"GRE", "SAT", "ACT"}


def _curated(lang):
    """Committed single-token lists, so tests are deterministic without the downloaded lists."""
    known = Vocabulary.from_csv(VOCAB_DIR / lang / "baseline.csv")
    advanced = Vocabulary.from_csv(VOCAB_DIR / lang / "advanced.csv")
    return known, advanced


def _exam_targets():
    """GRE/SAT/ACT words from the committed English advanced sample (by tier label)."""
    advanced = Vocabulary.from_csv(VOCAB_DIR / "en" / "advanced.csv")
    return sorted(w for w, tier in advanced.levels.items() if tier in _EXAM_TIERS)


def _run(lang, known, pool, *, seed):
    """Sample scenarios from (known, pool) and run the full eval with the mock generator."""
    scenarios = sample_scenarios(4, language=lang, seed=seed, known=known, target_pool=pool)
    summary = evaluate(
        f"mock-{lang}",
        scenarios,
        api_generator(mock=True),
        lemmatizer=get_analyzer(lang),
    )
    return scenarios, summary


def test_english_cefr_wordlist_eval():
    known, advanced = _curated("en")
    pool = sorted(advanced.lemmas - known.lemmas)
    scenarios, summary = _run("en", known, pool, seed=11)
    agg = summary.aggregate()
    assert agg["n"] == 4
    for s in scenarios:
        assert not (s.target_set() & s.known_set())  # targets genuinely new


def test_english_exam_wordlist_eval():
    """REQUIRED: GRE/SAT/ACT exam words as the to-learn targets, run through the eval."""
    known, _ = _curated("en")
    exam = _exam_targets()
    assert len(exam) >= 10, "exam words must be present in the curated advanced list"

    # Targets are drawn only from the exam set here.
    scenarios, summary = _run("en", known, exam, seed=7)
    agg = summary.aggregate()
    assert agg["n"] == 4

    exam_set = set(exam)
    used = {w for s in scenarios for w in s.target_set()}
    assert used, "no targets sampled"
    assert used <= exam_set  # every target is an exam word
    for s in scenarios:
        assert not (s.target_set() & s.known_set())  # exam words are never already known
    # The mock builds spec-passing stories, so exam targets clear the deterministic checks.
    assert summary.aggregate()["hard_pass_rate"] == 1.0


def test_english_exam_words_are_genuinely_hard():
    """Sanity: the exam set includes classic hard words, not easy A2 nouns."""
    exam = set(_exam_targets())
    assert {"vigilant", "voracious", "serpentine"} <= exam


@pytest.mark.parametrize(("lang", "pkg"), [("zh", "jieba"), ("ja", "fugashi")])
def test_cjk_wordlist_eval(lang, pkg):
    pytest.importorskip(pkg)
    known, advanced = _curated(lang)
    pool = sorted(advanced.lemmas - known.lemmas)
    scenarios, summary = _run(lang, known, pool, seed=3)
    assert summary.aggregate()["n"] == 4
    for s in scenarios:
        assert s.language == lang
        assert not (s.target_set() & s.known_set())
