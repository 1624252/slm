"""Results-over-time tracker: record extraction, history round-trip, and leaderboard rendering.

Deterministic — passes explicit timestamp/commit so it never shells out to git or the clock.
"""

import json

from islm.eval.track import (
    append_run,
    leaderboard_markdown,
    load_runs,
    record_from_results,
    track_results,
    write_leaderboard,
)

# A minimal eval payload shaped like what islm.eval.run writes.
PAYLOAD = {
    "base": {
        "model": "SmolLM2-135M",
        "n": 8,
        "hard_pass_rate": 0.0,
        "mean_oov_rate": 0.413,
        "one_new_word_pass_rate": 0.0,
        "recurrence_pass_rate": 0.125,
    },
    "tuned": {
        "model": "SmolLM2-135M+lora",
        "n": 8,
        "hard_pass_rate": 0.25,
        "mean_oov_rate": 0.30,
        "one_new_word_pass_rate": 0.125,
        "recurrence_pass_rate": 0.5,
    },
    "adversarial": {
        "base": {"hard_pass_rate": 0.0},
        "tuned": {"hard_pass_rate": 0.1},
    },
}


def _record(**kwargs):
    defaults = dict(
        label="r1", language="en", commit="abc123", timestamp="2026-07-08T00:00:00+00:00"
    )
    return record_from_results(PAYLOAD, **{**defaults, **kwargs})


def test_record_from_results_extracts_metrics_and_context():
    rec = _record(dataset="data/curated/seed", epochs=3)
    assert rec.base["hard_pass_rate"] == 0.0
    assert rec.tuned["hard_pass_rate"] == 0.25
    assert rec.n_scenarios == 8
    assert rec.base_model == "SmolLM2-135M"  # falls back to the payload's model name
    assert rec.tuned_model == "SmolLM2-135M+lora"
    assert rec.adversarial == {"base": 0.0, "tuned": 0.1}
    assert rec.epochs == 3
    assert rec.commit == "abc123"


def test_missing_judge_and_inferability_are_omitted():
    rec = _record()
    assert "judge_overall" not in rec.base and "mean_inferability" not in rec.tuned


def test_append_and_load_runs_roundtrip(tmp_path):
    runs = tmp_path / "runs.jsonl"
    append_run(_record(label="first", timestamp="2026-07-08T00:00:00+00:00"), runs)
    append_run(_record(label="second", timestamp="2026-07-09T00:00:00+00:00"), runs)
    loaded = load_runs(runs)
    assert [r["label"] for r in loaded] == ["first", "second"]  # append order preserved
    assert loaded[0]["base"]["hard_pass_rate"] == 0.0


def test_leaderboard_orders_newest_first_and_shows_delta(tmp_path):
    runs = tmp_path / "runs.jsonl"
    append_run(_record(label="older", timestamp="2026-07-01T00:00:00+00:00"), runs)
    append_run(_record(label="newer", timestamp="2026-07-05T00:00:00+00:00"), runs)
    md = leaderboard_markdown(load_runs(runs))
    assert md.index("newer") < md.index("older")  # newest first
    assert "0.000->0.250 (+0.250)" in md  # base->tuned (delta) for hard-pass
    assert "0.413->0.300 (-0.113)" in md  # OOV improved (negative delta)
    assert "Hard-pass" in md and "Commit" in md


def test_leaderboard_empty_is_graceful():
    assert "No runs recorded yet" in leaderboard_markdown([])


def test_base_only_run_renders_without_tuned(tmp_path):
    payload = {"base": PAYLOAD["base"]}  # e.g. a Day-1 baseline (no tuned model)
    rec = record_from_results(payload, label="baseline", language="en", commit="x", timestamp="t")
    md = leaderboard_markdown([json.loads(json.dumps(rec.__dict__))])
    assert "0.000" in md  # base value shown, no arrow / delta
    assert "->" not in md.split("baseline")[1].split("\n")[0]


def test_track_results_writes_history_and_leaderboard(tmp_path):
    results = tmp_path / "results_en.json"
    results.write_text(json.dumps(PAYLOAD), encoding="utf-8")
    runs, board = tmp_path / "runs.jsonl", tmp_path / "LEADERBOARD.md"
    rec = track_results(
        results, label="day3", dataset="data/curated/seed", runs_path=runs, leaderboard_path=board
    )
    assert rec.language == "en"  # inferred from results_en.json
    assert runs.exists() and board.exists()
    assert len(load_runs(runs)) == 1
    assert "day3" in board.read_text(encoding="utf-8")


def test_write_leaderboard_from_history(tmp_path):
    runs = tmp_path / "runs.jsonl"
    append_run(_record(), runs)
    out = write_leaderboard(runs, tmp_path / "board.md")
    assert out.exists() and "Results leaderboard" in out.read_text(encoding="utf-8")
