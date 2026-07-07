"""End-to-end smoke test: the mock teacher produces spec-compliant data and the eval runs.

Uses SimpleLemmatizer + MockLLM so it needs no network, API key, or spaCy model.
"""

from islm.datagen.generate import generate_story, make_example
from islm.datagen.scenarios import sample_scenarios
from islm.eval.harness import evaluate
from islm.eval.judge import judge_story
from islm.llm.client import MockLLM
from islm.vocab.lemmatize import SimpleLemmatizer

LEM = SimpleLemmatizer()


def test_mock_generation_is_spec_compliant():
    client = MockLLM()
    for scenario in sample_scenarios(4, language="en", seed=1):
        ex = make_example(
            scenario,
            client,
            lemmatizer=LEM,
            judge_fn=lambda s, story: judge_story(s, story, client),
        )
        assert ex.report.hard_pass, ex.report.failures()
        assert ex.kept
        for word in scenario.target_set():
            assert ex.report.recurrence.counts[word] >= 3
        record = ex.to_record("train")
        assert record["messages"][-1]["role"] == "assistant"
        assert record["metadata"]["hard_pass"] is True


def test_eval_harness_runs_with_mock():
    client = MockLLM()
    scenarios = sample_scenarios(4, language="en", seed=2)
    summary = evaluate(
        "mock",
        scenarios,
        lambda s: generate_story(s, client, temperature=0.0),
        judge_client=client,
        lemmatizer=LEM,
    )
    agg = summary.aggregate()
    assert agg["n"] == 4
    assert 0.0 <= agg["hard_pass_rate"] <= 1.0
    assert agg["judge_overall"] == 2.0  # mock judge gives full marks
