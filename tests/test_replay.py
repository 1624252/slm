"""Replay harness — re-scoring stored stories reproduces the recorded metrics, no model.

Builds a tiny results-shaped payload from known-good stories and confirms replay recomputes the
same hard_pass the harness would, entirely offline (deterministic validators only).
"""

import json

from islm.datagen.scenarios import Scenario, save_scenarios
from islm.eval.replay import replay, summarize


def _story(target="clue"):
    # Known-only frame + target introduced once then repeated (spec-passing by construction).
    known = ["the", "cat", "is", "big", "see", "a"]
    return known, target


def test_replay_reproduces_hard_pass(tmp_path):
    known, target = _story()
    scn = Scenario("en-0000", "en", "A1-A2", "t", [target], known)
    scenarios_file = tmp_path / "heldout_en.jsonl"
    save_scenarios([scn], scenarios_file)

    story = "The cat is big. The cat see a clue. The clue is big. The clue is big. The clue is big."
    payload = {
        "tuned_rows": [{"id": "en-0000", "story": story, "hard_pass": True, "oov_rate": 0.0}],
    }
    results = tmp_path / "results_en.json"
    results.write_text(json.dumps(payload), encoding="utf-8")

    out = replay(results, "en", side="tuned", curated=False, scenarios_path=scenarios_file)
    assert out["tuned"]["n"] == 1
    assert out["tuned"]["hard_pass_rate"] == 1.0
    assert out["tuned"]["matches_recorded"] is True


def test_replay_flags_mismatch(tmp_path):
    """If a recorded hard_pass disagrees with re-scoring, replay reports matches_recorded=False."""
    known, target = _story()
    scn = Scenario("en-0000", "en", "A1-A2", "t", [target], known)
    scenarios_file = tmp_path / "heldout_en.jsonl"
    save_scenarios([scn], scenarios_file)

    story = "The cat is big. The cat see a clue. The clue is big. The clue is big. The clue is big."
    # Record a WRONG hard_pass=False for a story that actually passes.
    payload = {"tuned_rows": [{"id": "en-0000", "story": story, "hard_pass": False}]}
    results = tmp_path / "results_en.json"
    results.write_text(json.dumps(payload), encoding="utf-8")

    out = replay(results, "en", side="tuned", curated=False, scenarios_path=scenarios_file)
    assert out["tuned"]["matches_recorded"] is False


def test_summarize_empty():
    assert summarize([]) == {"n": 0}
