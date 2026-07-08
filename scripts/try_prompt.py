"""Run ONE custom prompt through the model and save the output.

Not an eval — just "prompt in, story out" for a single scenario. Word lists can come from
several sources so you rarely have to type them by hand:

  * exam-level tiers   --known-tier A1,A2   --target-tier B1
                       (CEFR for en, HSK for zh, JLPT for ja — read from data/vocab/<lang>/*.csv)
  * an existing scenario  --from-scenario evals/scenarios/heldout_small_en.jsonl --index 0
  * a plain file          --known-file my_words.txt   (one word per line, or comma-separated)
  * inline text           --known "the,a,is,dog,cat"  --target fox

Precedence per field (known / target): --from-scenario < tier < file < inline. So you can load
a scenario, then override just the target word inline, etc.

Usage (from repo root; drop PYTHONPATH once you've run `pip install -e .`):

    # Known = CEFR A1+A2, target = a sampled B1 word (English):
    PYTHONPATH=src python scripts/try_prompt.py --adapter outputs/day3_lora \
        --known-tier A1,A2 --target-tier B1 --known-limit 150 \
        --theme "a walk in the park" --out outputs/try/tier.json

    # Chinese, HSK: known = HSK1-3, target = an HSK4 word:
    PYTHONPATH=src python scripts/try_prompt.py --language zh \
        --known-tier HSK1,HSK2,HSK3 --target-tier HSK4

    # Reuse a held-out test scenario verbatim:
    PYTHONPATH=src python scripts/try_prompt.py --adapter outputs/day3_lora \
        --from-scenario evals/scenarios/heldout_small_en.jsonl --index 2

    # Fully hand-typed:
    PYTHONPATH=src python scripts/try_prompt.py --target fox \
        --known "the,a,is,dog,cat,see,run,big,park" --theme "a walk in the park"

Omit --adapter to run the plain base model. --list-tiers / --list-scenarios inspect and exit.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from islm.datagen.scenarios import Scenario, load_scenarios
from islm.eval.generators import HFGenerator
from islm.llm.prompts import generation_prompt
from islm.vocab.wordlists import VOCAB_DIR, Vocabulary


def _split_words(text: str) -> list[str]:
    """Split on commas / whitespace / newlines; drop blanks. Works for inline args and files."""
    return [w for w in re.split(r"[,\s]+", text.strip()) if w]


def _tier_index(language: str) -> dict[str, str]:
    """Map every graded word for a language to its exam tier (CEFR / HSK / JLPT).

    Reads all committed tier files (curated + full, baseline + advanced) so the labels are the
    real exam levels stored in each CSV's `tier` column.
    """
    vocab = Vocabulary()
    for name in ("baseline.full.csv", "baseline.csv", "advanced.full.csv", "advanced.csv"):
        path = VOCAB_DIR / language / name
        if path.exists():
            vocab = vocab | Vocabulary.from_csv(path)
    return vocab.levels


def _tiers_available(language: str) -> list[str]:
    """Sorted distinct exam tiers present for a language (e.g. A1..C2, HSK1..HSK7-9, N5..N1)."""
    return sorted(set(_tier_index(language).values()))


def _tier_words(language: str, tiers: list[str]) -> list[str]:
    """Sorted words whose exam tier is in `tiers` (case-insensitive match)."""
    wanted = {t.upper() for t in tiers}
    index = _tier_index(language)
    return sorted(w for w, tier in index.items() if tier.upper() in wanted)


def _resolve_field(
    *, scenario_words, tiers, file_path, inline, language, limit, sample, seed
):
    """Apply the precedence scenario < tier < file < inline, then optionally sample/cap."""
    words = list(scenario_words or [])
    if tiers:
        words = _tier_words(language, tiers)
    if file_path:
        words = _split_words(Path(file_path).read_text(encoding="utf-8"))
    if inline:
        words = _split_words(inline)
    if sample and len(words) > sample:
        words = random.Random(seed).sample(words, sample)
    if limit is not None and len(words) > limit:
        words = words[:limit]
    return words


def _load_from_scenario(path: Path, index: int) -> Scenario:
    scenarios = load_scenarios(path)
    if not scenarios:
        raise SystemExit(f"{path}: no scenarios found")
    if not -len(scenarios) <= index < len(scenarios):
        raise SystemExit(f"{path}: index {index} out of range (has {len(scenarios)})")
    return scenarios[index]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run one custom prompt through the model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model
    p.add_argument("--base", default="HuggingFaceTB/SmolLM2-135M-Instruct", help="Base HF model.")
    p.add_argument("--adapter", default=None, help="LoRA adapter dir (omit for plain base).")
    # Scenario framing
    p.add_argument("--language", default="en")
    p.add_argument("--level", default="A1-A2")
    p.add_argument("--theme", default="a walk in the park")
    # Load a whole scenario from a file (lowest precedence for word lists)
    p.add_argument(
        "--from-scenario", type=Path, default=None, help="JSONL of scenarios to seed fields from."
    )
    p.add_argument("--index", type=int, default=0, help="Which scenario in --from-scenario.")
    p.add_argument(
        "--list-scenarios", action="store_true", help="Print --from-scenario contents and exit."
    )
    p.add_argument(
        "--list-tiers", action="store_true", help="Print this language's exam tiers and exit."
    )
    # Known words: exam tier(s) | file | inline
    p.add_argument(
        "--known-tier", default=None, help="Exam tier(s) for known words, e.g. A1,A2 / HSK1,HSK2."
    )
    p.add_argument("--known-file", default=None, help="File of known words (comma/space/newline).")
    p.add_argument("--known", default=None, help="Inline comma-separated known words.")
    p.add_argument(
        "--known-limit", type=int, default=200, help="Cap known-list size (keeps CPU prompts sane)."
    )
    # Target words: exam tier(s) sampled | file | inline
    p.add_argument(
        "--target-tier", default=None, help="Exam tier(s) to sample target (new) words from."
    )
    p.add_argument("--target-file", default=None, help="File of target words.")
    p.add_argument("--target", default=None, help="Inline comma-separated target (new) words.")
    p.add_argument("--n-targets", type=int, default=1, help="How many to sample with a tier.")
    # Generation / output
    p.add_argument("--seed", type=int, default=0, help="Seed for target/known sampling.")
    p.add_argument("--max-new-tokens", type=int, default=160)
    p.add_argument("--out", type=Path, default=Path("outputs/try/prompt.json"))
    args = p.parse_args()

    if args.list_tiers:
        tiers = _tiers_available(args.language)
        print(f"{args.language}: {', '.join(tiers) if tiers else '(none — frequency fallback)'}")
        return
    if args.list_scenarios:
        if not args.from_scenario:
            raise SystemExit("--list-scenarios needs --from-scenario <file>")
        for i, s in enumerate(load_scenarios(args.from_scenario)):
            summary = f"{s.language} | {s.theme} | target={s.target_words} | known={len(s.known)}"
            print(f"[{i}] {summary}")
        return

    seed_scn = _load_from_scenario(args.from_scenario, args.index) if args.from_scenario else None
    known_tiers = _split_words(args.known_tier) if args.known_tier else None
    target_tiers = _split_words(args.target_tier) if args.target_tier else None

    known = _resolve_field(
        scenario_words=seed_scn.known if seed_scn else None,
        tiers=known_tiers,
        file_path=args.known_file,
        inline=args.known,
        language=args.language,
        limit=args.known_limit,
        sample=None,
        seed=args.seed,
    )
    target = _resolve_field(
        scenario_words=seed_scn.target_words if seed_scn else None,
        tiers=target_tiers,
        file_path=args.target_file,
        inline=args.target,
        language=args.language,
        limit=None,
        sample=args.n_targets if target_tiers else None,
        seed=args.seed,
    )
    if not known:
        known = _split_words("the,a,is,big,small,dog,cat,see,run,happy,in,park")
    if not target:
        target = ["fox"]

    scenario = Scenario(
        id="custom-0001",
        language=seed_scn.language if seed_scn else args.language,
        level=seed_scn.level if seed_scn else args.level,
        theme=seed_scn.theme if seed_scn else args.theme,
        target_words=target,
        known=known,
    )

    system, user = generation_prompt(scenario)
    gen = HFGenerator(args.base, adapter_path=args.adapter, max_new_tokens=args.max_new_tokens)
    story = gen(scenario)

    print("=" * 70)
    print(user)
    print("=" * 70)
    print(story)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "base_model": args.base,
        "adapter": args.adapter,
        "scenario": scenario.to_dict(),
        "prompt": {"system": system, "user": user},
        "story": story,
    }
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
