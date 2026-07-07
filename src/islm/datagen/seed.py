"""A small, human-authored seed of genuinely good English i+1 stories.

Automated generation needs a teacher model (an API key or local server). Where that isn't
available, this module still produces *real, spec-passing* data — hand-authored to the Behavior
Spec — so we can (a) show what good data looks like and (b) exercise the second-pass curation on
real prose rather than mock filler.

    python -m islm.datagen.seed --out data/generated/en_seed

Each story is validated against the real English baseline vocabulary; only spec-passing stories
are written. One near-duplicate is included on purpose to demonstrate curation dedup.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import DATA_DIR
from ..validators import validate_story
from ..vocab.lemmatize import get_analyzer
from ..vocab.wordlists import load_baseline
from .generate import Example
from .scenarios import Scenario

# (target_words, story). Stories use only common A1-A2 vocabulary plus the single target word,
# introduce the target once, and repeat it for spaced repetition.
SEED_STORIES: list[tuple[list[str], str]] = [
    (
        ["shadow"],
        "The girl has a little cat. The cat is black. "
        "At night, the cat sees a shadow on the box. "
        "The shadow is very big. The cat looks at the shadow. "
        "The shadow looks at the cat too. "
        "But the shadow is only the little cat! "
        "Now the girl and the cat play with the shadow every night.",
    ),
    (
        ["treasure"],
        "The girl and her dog go to the garden. "
        "The dog looks and looks in the garden. "
        "Then the dog finds a little box. "
        "In the box is a treasure! "
        "The treasure is old and red. "
        "The girl is so happy with the treasure. "
        "She takes the treasure home. "
        "The dog wants the treasure too.",
    ),
    (
        ["secret"],
        "The boy has a secret. He does not tell his friend. "
        "The secret is very big. "
        "The boy wants to tell the secret, but he waits. "
        "At night, he tells the cat the secret. "
        "The cat likes the secret. "
        "Now the cat and the boy have a secret.",
    ),
    (
        ["umbrella"],
        "It is a cold day. The rain comes down. "
        "The girl has a new umbrella. The umbrella is blue. "
        "She walks in the rain with the umbrella. "
        "A little bird is cold. "
        "The girl gives the bird her umbrella. "
        "Now the bird is happy under the umbrella.",
    ),
    (
        ["feather"],
        "A little bird sits in the tree. "
        "The bird has one red feather. "
        "The feather falls down to the garden. "
        "A cat sees the feather and looks up. "
        "The bird wants the feather back. "
        "The cat gives the feather to the bird.",
    ),
    (
        ["lantern"],
        "It is night and the house is dark. "
        "The old man has a lantern. "
        "He looks for the cat with the lantern. "
        "The lantern is not very big, but it helps. "
        "With the lantern, the man finds the little cat. "
        "The cat sleeps by the warm lantern.",
    ),
]

# Deliberate near-duplicate of the first story (curation should drop it).
_DUPLICATE_OF = 0


def build(out_dir: Path) -> dict:
    known = load_baseline("en")
    analyzer = get_analyzer("en")
    kept: list[dict] = []
    failed: list[dict] = []

    stories = list(SEED_STORIES) + [SEED_STORIES[_DUPLICATE_OF]]
    for i, (targets, story) in enumerate(stories):
        scenario = Scenario(
            id=f"en-seed-{i:03d}",
            language="en",
            level="A1-A2",
            theme="seed",
            target_words=targets,
            known=sorted(known.lemmas),
        )
        report = validate_story(story, known.lemmas, set(targets), analyzer, language="en")
        example = Example(scenario, story, report, rewrite_passes=0, kept=report.hard_pass)
        record = example.to_record("train")
        (kept if report.hard_pass else failed).append(
            record if report.hard_pass else {**record, "failures": report.failures()}
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "train.jsonl", "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for name in ("val", "test"):  # keep the split layout the curator expects
        (out_dir / f"{name}.jsonl").write_text("", encoding="utf-8")

    return {"authored": len(stories), "spec_passing": len(kept), "failed": failed}


def main() -> None:
    p = argparse.ArgumentParser(description="Build the human-authored English seed dataset.")
    p.add_argument("--out", type=Path, default=DATA_DIR / "generated" / "en_seed")
    args = p.parse_args()
    stats = build(args.out)
    print(f"authored={stats['authored']} spec_passing={stats['spec_passing']}")
    for bad in stats["failed"]:
        print("FAILED", bad["id"], bad.get("failures"))


if __name__ == "__main__":
    main()
