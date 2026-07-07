"""End-to-end dataset build: sample scenarios -> generate + filter -> write JSONL splits.

Run offline (no API key needed) with the mock teacher:

    python -m islm.datagen.pipeline --n 20 --mock --out data/generated

Or with a real teacher/judge (set keys in .env):

    python -m islm.datagen.pipeline --n 500 --model gpt-5 --judge-model claude-sonnet-5
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from ..config import DATA_DIR, DEFAULT_THRESHOLDS, Thresholds
from ..llm.client import LLMClient, get_client
from ..vocab.lemmatize import get_analyzer
from .generate import Example, make_example
from .scenarios import sample_scenarios


def _assign_splits(
    examples: list[Example], splits: tuple[float, float, float], seed: int
) -> dict[str, list[Example]]:
    kept = [e for e in examples if e.kept]
    random.Random(seed).shuffle(kept)  # scenario-level split (each example is one scenario)
    n = len(kept)
    n_train = int(n * splits[0])
    n_val = int(n * splits[1])
    return {
        "train": kept[:n_train],
        "val": kept[n_train : n_train + n_val],
        "test": kept[n_train + n_val :],
    }


def _summarize(examples: list[Example]) -> dict:
    n = len(examples)
    kept = [e for e in examples if e.kept]

    def mean(values: list[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    return {
        "scenarios": n,
        "kept": len(kept),
        "keep_rate": round(len(kept) / n, 4) if n else 0.0,
        "mean_rewrite_passes": mean([e.rewrite_passes for e in examples]),
        "mean_oov_rate": mean([e.report.coverage.oov_rate for e in examples]),
        "mean_coverage": mean([e.report.coverage.coverage for e in examples]),
        "one_new_word_pass_rate": mean([float(e.report.one_new_word.passed) for e in examples]),
        "recurrence_pass_rate": mean([float(e.report.recurrence.passed) for e in examples]),
        "hard_pass_rate": mean([float(e.report.hard_pass) for e in examples]),
    }


def build_dataset(
    n: int = 50,
    language: str = "en",
    seed: int = 0,
    out_dir: Path = DATA_DIR / "generated",
    client: LLMClient | None = None,
    judge_client: LLMClient | None = None,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    max_rewrites: int = 5,
    splits: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict:
    """Build and write the dataset; return summary stats."""
    if client is None:
        client = get_client()
    lemmatizer = get_analyzer(language)

    judge_fn = None
    if judge_client is not None:
        from ..eval.judge import judge_story  # local import avoids an import cycle

        def judge_fn(scenario, story):  # noqa: E731 - small closure over the judge client
            return judge_story(scenario, story, judge_client)

    scenarios = sample_scenarios(n, language=language, seed=seed)
    examples = [
        make_example(s, client, lemmatizer, thresholds, max_rewrites, judge_fn) for s in scenarios
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    by_split = _assign_splits(examples, splits, seed)
    for split, items in by_split.items():
        with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for e in items:
                f.write(json.dumps(e.to_record(split), ensure_ascii=False) + "\n")

    stats = _summarize(examples)
    stats["split_counts"] = {k: len(v) for k, v in by_split.items()}
    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description="Build the i+1 story dataset.")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--language", default="en", help="Language code, e.g. en, zh, ja.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=DATA_DIR / "generated")
    p.add_argument("--model", default=None, help="Teacher model (defaults to TEACHER_MODEL).")
    p.add_argument("--judge-model", default=None, help="Judge model; omit to skip judging.")
    p.add_argument("--max-rewrites", type=int, default=5)
    p.add_argument("--mock", action="store_true", help="Use the offline MockLLM (no API key).")
    args = p.parse_args()

    client = get_client(args.model, mock=args.mock)
    judge_client = None
    if args.mock:
        judge_client = client  # mock also serves canned judgements
    elif args.judge_model:
        judge_client = get_client(args.judge_model)

    stats = build_dataset(
        n=args.n,
        language=args.language,
        seed=args.seed,
        out_dir=args.out,
        client=client,
        judge_client=judge_client,
        max_rewrites=args.max_rewrites,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
