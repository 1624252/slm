"""Teacher-regen generator — offline (MockLLM). Records are well-formed, spec-passing, and the
judge gate drops low-coherence stories. No network/key needed."""

import json

import pytest

from islm.datagen.teacher import TeacherConfig, _judge_gate, generate
from islm.llm.client import MockLLM


@pytest.mark.parametrize("lang", ["en", "zh", "ja"])
def test_generate_produces_spec_passing_records(tmp_path, lang):
    mock = MockLLM()
    stats = generate(
        lang, n=6, out_dir=tmp_path / lang, client=mock, judge_client=mock, seed=1
    )
    assert stats["kept"] == 6
    assert stats["source"] == "teacher-v2"
    recs = [
        json.loads(line)
        for line in (tmp_path / lang / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for r in recs:
        assert [m["role"] for m in r["messages"]] == ["system", "user", "assistant"]
        assert r["metadata"]["hard_pass"] is True  # only kept records are written
        assert r["metadata"]["source"] == "teacher-v2"
        assert r["target_words"]


def test_no_judge_path_keeps_deterministic(tmp_path):
    """Without a judge client the gate is skipped; deterministic hard-pass still governs keeping."""
    stats = generate(
        "en", n=4, out_dir=tmp_path / "en", client=MockLLM(), judge_client=None, seed=2
    )
    assert stats["kept"] == 4


def test_judge_gate_rejects_low_coherence():
    cfg = TeacherConfig()
    good = {"coherence": 2, "task_quality": 2, "interestingness": 1}
    flat = {"coherence": 0, "task_quality": 1, "interestingness": 0}  # the gamed-text failure mode
    partial = {"coherence": 2, "task_quality": 1, "interestingness": 2}  # task_quality too low
    assert _judge_gate(good, cfg) is True
    assert _judge_gate(flat, cfg) is False
    assert _judge_gate(partial, cfg) is False
    assert _judge_gate(None, cfg) is True  # no judge -> gate is a no-op


def test_splits_are_disjoint(tmp_path):
    generate("en", n=10, out_dir=tmp_path / "en", client=MockLLM(), judge_client=MockLLM(), seed=3)
    ids = set()
    for split in ("train", "val", "test"):
        path = tmp_path / "en" / f"{split}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            rid = json.loads(line)["id"]
            assert rid not in ids, "record id appears in more than one split"
            ids.add(rid)
