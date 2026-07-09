"""Synthetic generator — every produced story is spec-passing and grammatically framed (offline)."""

import json

import pytest

from islm.datagen.synth import TARGET_POOLS, generate


@pytest.mark.parametrize("lang", ["en", "zh", "ja"])
def test_generate_produces_spec_passing_records(tmp_path, lang):
    out = tmp_path / lang
    stats = generate(lang, n=30, out_dir=out, seed=1)
    assert stats["kept"] == 30
    assert stats["failed"] == 0  # POS-routed frames + scoped known set => always spec-passing
    lines = (out / "train.jsonl").read_text(encoding="utf-8").splitlines()
    recs = [json.loads(line) for line in lines]
    assert recs
    for r in recs:
        assert [m["role"] for m in r["messages"]] == ["system", "user", "assistant"]
        assert r["metadata"]["hard_pass"] is True
        assert r["metadata"]["source"] == "synthetic-v1"


def test_english_targets_are_pos_typed():
    pools = TARGET_POOLS["en"]
    assert pools["noun"] and pools["adj"] and pools["verb"]
    # a known concrete noun, adjective, and verb are present where expected
    assert "beacon" in pools["noun"]
    assert "radiant" in pools["adj"]
    assert "wander" in pools["verb"]


def test_english_uses_articles_correctly():
    """No 'a apple' / 'a ominous' — vowel-initial targets take 'an'."""
    from islm.datagen.synth import _a

    assert _a("apple") == "an" and _a("ominous") == "an"
    assert _a("beacon") == "a" and _a("wolf") == "a"
