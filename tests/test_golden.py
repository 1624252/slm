"""Golden set — the Layer-1 regression gate: every committed reference story MUST pass.

If any of these fail, something is fundamentally broken (a validator regressed, or the data was
edited). This is the "run on every commit, all must pass" set from the 6-layer eval model. Runs
offline against the deterministic validators — no model, no network.
"""

import json

import pytest

from islm.datagen.seed import SEED
from islm.eval.golden import GOLD, build_items
from islm.validators import validate_story
from islm.vocab.lemmatize import get_analyzer

# Build once (validates on build); reuse across tests.
_ITEMS, _STATS = build_items(list(GOLD))


def test_golden_set_size_in_range():
    # PDF says start at 10-20; the user asked for 50-200. We keep >=50.
    assert 50 <= len(_ITEMS) <= 200, f"golden set has {len(_ITEMS)} items"


def test_every_gold_story_hard_passes():
    """The defining property: each reference output satisfies all deterministic checks."""
    for rec in _ITEMS:
        assert rec["metadata"]["hard_pass"] is True, rec["id"]


def test_gold_records_have_required_metadata():
    for rec in _ITEMS:
        assert set(rec.keys()) >= {"id", "language", "target_words", "messages", "metadata"}
        m = rec["metadata"]
        for key in ("tone", "keywords", "target_tier", "category", "difficulty", "source"):
            assert key in m, f"{rec['id']} missing metadata.{key}"
        assert m["source"] == "golden-authored"
        # three-message chat: system rules, user scenario, assistant reference story
        roles = [msg["role"] for msg in rec["messages"]]
        assert roles == ["system", "user", "assistant"]


def test_gold_is_held_out_from_training_seed():
    """No leakage: no gold story text appears in the training seed (SEED)."""
    seed_stories = {story for items in SEED.values() for _, story in items}
    gold_stories = {rec["messages"][2]["content"] for rec in _ITEMS}
    assert not (seed_stories & gold_stories), "a gold story is reused from the training seed"


def test_gold_covers_all_languages():
    langs = {rec["language"] for rec in _ITEMS}
    assert langs == {"en", "zh", "ja"}


def test_gold_includes_exam_tier_targets():
    """Layer-2 coverage: the hard exam tier is represented among gold targets."""
    tiers = {t for rec in _ITEMS for t in rec["metadata"]["target_tier"]}
    assert tiers & {"GRE", "SAT", "ACT"}


@pytest.mark.parametrize("lang", ["en", "zh", "ja"])
def test_gold_revalidates_from_stored_scenario(lang):
    """Re-run the validator on each story using its own known/target — must still hard-pass."""
    analyzer = get_analyzer(lang)
    items = [r for r in _ITEMS if r["language"] == lang]
    assert items
    for rec in items:
        story = rec["messages"][2]["content"]
        # KNOWN_WORDS / TARGET_WORDS are parseable from the user message.
        user = rec["messages"][1]["content"]
        known = _parse_field(user, "KNOWN_WORDS")
        target = _parse_field(user, "TARGET_WORDS")
        report = validate_story(story, set(known), set(target), analyzer, language=lang)
        assert report.hard_pass, f"{rec['id']}: {report.failures()}"


def _parse_field(user: str, label: str) -> list[str]:
    for line in user.splitlines():
        if line.startswith(label + ":"):
            return [w.strip().lower() for w in line.split(":", 1)[1].split(",") if w.strip()]
    return []


def test_golden_jsonl_on_disk_matches_build(tmp_path):
    """The committed golden.jsonl parses and every line is a hard-passing record."""
    from islm.config import EVALS_DIR

    path = EVALS_DIR / "golden" / "golden.jsonl"
    if not path.exists():
        pytest.skip("golden.jsonl not built on disk")
    disk = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    assert len(disk) == len(_ITEMS)
    assert all(r["metadata"]["hard_pass"] for r in disk)
