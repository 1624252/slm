"""LangSmith sync — offline conversion tests only.

The actual push is gated on LANGSMITH_API_KEY and never runs here; we test the pure conversion
from our record schemas into LangSmith's {inputs, outputs, metadata} shape.
"""

import json

from islm.eval.golden import GOLD, build
from islm.eval.langsmith_sync import golden_to_examples, push_golden, results_to_records


def test_golden_to_examples_shape(tmp_path):
    out = tmp_path / "golden"
    build(out, list(GOLD))
    examples = golden_to_examples(out / "golden.jsonl")
    assert len(examples) >= 50
    e = examples[0]
    assert set(e["inputs"]) == {"language", "target_words", "system", "user"}
    assert "reference_story" in e["outputs"] and e["outputs"]["hard_pass"] is True
    assert e["metadata"]["source"] == "golden-authored"
    assert e["metadata"]["category"] and e["metadata"]["difficulty"]


def test_push_golden_is_noop_without_key(tmp_path, monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    out = tmp_path / "golden"
    build(out, list(GOLD))
    # No key -> dry run -> returns 0, never touches the network.
    assert push_golden(out / "golden.jsonl") == 0


def test_results_to_records(tmp_path):
    row = {"id": "en-0000", "story": "s", "hard_pass": False, "oov_rate": 0.1, "failures": ["oov"]}
    payload = {"tuned_rows": [row]}
    path = tmp_path / "results_en.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    recs = results_to_records(path)
    assert recs[0]["id"] == "en-0000" and recs[0]["failures"] == ["oov"]
