"""Evaluation: judge parsing, aggregation, robustness, report, and model generators.

Uses MockLLM / fakes so it is deterministic and needs no network or model download.
"""

from islm.datagen.scenarios import Scenario, sample_scenarios
from islm.eval import (
    api_generator,
    error_analysis,
    evaluate,
    guarded,
    judge_story,
    make_adversarial_scenarios,
    results_markdown,
)
from islm.eval.judge import DIMENSIONS, SPEC_DIMENSIONS
from islm.llm.client import MockLLM
from islm.vocab.lemmatize import SimpleLemmatizer

LEM = SimpleLemmatizer()


class _FakeClient:
    def __init__(self, text):
        self.text = text

    def complete(self, system, user, *, temperature=0.0, max_tokens=1024):
        return self.text


def _scenario(sid="s", targets=("clue",), known=None):
    known = known or ["the", "cat", "see", "is", "be", "big", "a", "clue"]
    return Scenario(sid, "en", "A1-A2", "t", list(targets), known)


def test_spec_dimensions_match_appendix_a():
    assert SPEC_DIMENSIONS == ("spec_adherence", "robustness", "task_quality", "consistency")
    assert all(d in DIMENSIONS for d in SPEC_DIMENSIONS)


def test_judge_parses_fenced_json_and_clamps():
    raw = '```json\n{"spec_adherence": 5, "robustness": -1, "task_quality": 1}\n```'
    scores = judge_story(_scenario(), "story", _FakeClient(raw))
    assert scores["spec_adherence"] == 2  # clamped to 2
    assert scores["robustness"] == 0  # clamped to 0
    assert scores["task_quality"] == 1
    assert scores["consistency"] == 0  # missing -> 0


def test_judge_handles_garbage():
    scores = judge_story(_scenario(), "story", _FakeClient("no json here"))
    assert all(scores[d] == 0 for d in DIMENSIONS)


def test_evaluate_and_aggregate_with_mock():
    scenarios = sample_scenarios(4, language="en", seed=5)
    summary = evaluate(
        "mock", scenarios, api_generator(mock=True), judge_client=MockLLM(), lemmatizer=LEM
    )
    agg = summary.aggregate()
    assert agg["n"] == 4
    assert "judge_spec_adherence" in agg and agg["judge_robustness"] == 2.0


def test_adversarial_targets_not_in_small_known():
    scenarios = make_adversarial_scenarios(4, language="en", seed=1)
    for s in scenarios:
        assert s.known and s.target_words
        assert not (s.target_set() & s.known_set())


def test_error_analysis_counts_failures():
    # A generator that always emits an OOV-heavy, too-short story -> failures.
    bad = evaluate("bad", [_scenario()], lambda s: "The dragon flew.", lemmatizer=LEM)
    ea = error_analysis(bad)
    assert ea["failing"] == 1
    assert "oov" in ea["reasons"]


def test_results_markdown_has_sections_and_verdict():
    scenarios = sample_scenarios(3, language="en", seed=2)
    base = evaluate(
        "base", scenarios, api_generator(mock=True), judge_client=MockLLM(), lemmatizer=LEM
    )
    tuned = evaluate(
        "tuned", scenarios, api_generator(mock=True), judge_client=MockLLM(), lemmatizer=LEM
    )
    md = results_markdown(base, tuned, base, tuned)
    assert "Behavioral checks" in md
    assert "LLM-as-judge rubric" in md
    assert "Robustness" in md
    assert "Win condition" in md
    assert "Error analysis" in md


def test_guard_recovers_a_bad_generation():
    scenario = _scenario()
    good = "The cat is big. The cat see a clue. The clue is big. The clue is big. The clue is big."
    gen = guarded(lambda s: "The dragon flew wildly.", lambda sc, st, f: good)
    result = gen(scenario)
    assert "clue" in result
