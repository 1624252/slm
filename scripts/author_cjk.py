"""Assemble teacher-quality v2 records from hand-authored (model-authored) CJK stories.

The English v2 set used an LLM teacher over the OpenAI API. For zh/ja we avoid that API entirely
(no contention with a running Colab eval): the stories are authored directly, then validated with
the *same* deterministic pipeline teacher.py uses — compact-known scoping (known = the story's own
content words minus targets) so coverage passes by construction, then the hard checks
(<=1-new-word/sentence, recurrence >=3x). Only hard-passing stories are kept and streamed to
`all.jsonl` in the dataset_v1/v2 record schema.

This is a *library*: `validate_batch()` scores a list of authored stories; callers feed batches in.
Run as a module to validate a JSON batch file:

    python -m scripts.author_cjk --language zh --in batch_zh.json --out data/generated/teacher_zh
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from islm.config import DEFAULT_THRESHOLDS
from islm.datagen.generate import Example
from islm.datagen.scenarios import Scenario
from islm.datagen.synth import _compact_known
from islm.validators import validate_story
from islm.vocab.lemmatize import get_analyzer


def build_example(language: str, targets: list[str], story: str, idx: int) -> Example:
    """Scope known to the story's own words (compact-known) and validate, mirroring teacher.py."""
    analyzer = get_analyzer(language)
    target_set = {t.lower() for t in targets}
    known = _compact_known(story, target_set, analyzer)
    scenario = Scenario(
        id=f"{language}-teacher-{idx:06d}",
        language=language,
        level="baseline",
        theme="",
        target_words=targets,
        known=sorted(known),
    )
    report = validate_story(
        story, set(known), target_set, analyzer, DEFAULT_THRESHOLDS, language=language
    )
    # No LLM judge here (no API); the deterministic hard-pass is the gate. Coherence is the
    # author's responsibility, verified by spot-review, not a judge call.
    return Example(scenario, story, report, rewrite_passes=0, judge=None, kept=report.hard_pass)


def validate_batch(language: str, items: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return (kept_records, rejects). Each item: {"targets": [...], "story": "..."}."""
    kept: list[dict] = []
    rejects: list[dict] = []
    for i, it in enumerate(items):
        ex = build_example(language, it["targets"], it["story"], len(kept))
        if ex.kept:
            rec = ex.to_record()
            rec["metadata"]["source"] = "teacher-v2-authored"
            kept.append(rec)
        else:
            rejects.append({"idx": i, "targets": it["targets"], "failures": ex.report.failures()})
    return kept, rejects


def append_records(out_dir: Path, records: list[dict]) -> None:
    """Append kept records to all.jsonl (crash-safe, matches teacher.py streaming)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "all.jsonl", "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Validate + assemble hand-authored CJK v2 stories.")
    p.add_argument("--language", required=True, choices=["zh", "ja"])
    p.add_argument("--in", dest="inp", type=Path, required=True, help="JSON list of {targets,story}.")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    items = json.loads(args.inp.read_text(encoding="utf-8"))
    kept, rejects = validate_batch(args.language, items)
    append_records(args.out, kept)
    print(f"[{args.language}] kept {len(kept)}/{len(items)}; rejected {len(rejects)}")
    for r in rejects:
        print(f"  reject #{r['idx']} targets={r['targets']} failures={r['failures']}")


if __name__ == "__main__":
    main()
