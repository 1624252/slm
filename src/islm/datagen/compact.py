"""Compact a dataset's KNOWN_WORDS to fit a small training window.

Teacher-generated records (`islm.datagen.pipeline`) embed the FULL baseline (~2.3k words) as
KNOWN_WORDS, so each rendered record is ~5k tokens. Training left-truncates to `--max-seq-len`,
which would chop the prompt and drop the task — the same bug the authored seed avoids via
`seed._compact_known`. This tool rewrites each record's KNOWN_WORDS to the compact set (curated
baseline ∪ the story's own content words, minus the target) and re-renders the prompt, so records
shrink to a few hundred tokens and match the seed/eval setup. Records whose story then still fails
validation are dropped (rare — the compact set covers the story's own words by construction).

    python -m islm.datagen.compact --in data/generated/teacher_en --out data/compact/teacher_en
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..validators import validate_story
from ..vocab.lemmatize import get_analyzer
from .generate import Example
from .scenarios import Scenario
from .seed import _compact_known


def compact_dir(in_dir: Path, out_dir: Path) -> dict:
    """Rewrite every split's records with compact KNOWN_WORDS; keep only still-passing ones."""
    out_dir.mkdir(parents=True, exist_ok=True)
    analyzers: dict[str, object] = {}
    stats: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        src = in_dir / f"{split}.jsonl"
        if not src.exists():
            continue
        kept, dropped = [], 0
        for line in src.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            lang = rec["language"]
            analyzer = analyzers.setdefault(lang, get_analyzer(lang))
            targets = rec["target_words"]
            story = rec["messages"][2]["content"]
            known = _compact_known(story, targets, lang, analyzer)
            scenario = Scenario(
                id=rec["id"], language=lang, level=rec.get("level", "baseline"),
                theme=rec.get("theme", ""), target_words=targets, known=known,
            )
            report = validate_story(story, set(known), set(targets), analyzer, language=lang)
            if not report.hard_pass:
                dropped += 1
                continue
            new_rec = Example(scenario, story, report, 0, kept=True).to_record(split=split)
            new_rec["metadata"]["source"] = "teacher-compacted"
            kept.append(new_rec)
        with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats[split] = {"kept": len(kept), "dropped": dropped}
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description="Compact KNOWN_WORDS so teacher records fit training.")
    p.add_argument("--in", dest="in_dir", type=Path, required=True)
    p.add_argument("--out", dest="out_dir", type=Path, required=True)
    args = p.parse_args()
    stats = compact_dir(args.in_dir, args.out_dir)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
