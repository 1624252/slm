"""Teacher-regenerated i+1 stories — the *quality* path (vs the meaning-blind templates in synth).

`synth.py` composes stories from fixed frames applied without regard to word meaning or narrative
flow ("To be forlorn is good.", "The scroll is a good friend."). It passes the deterministic
validators but reads as nonsense, and SFT on it taught the model to game the checks with dull,
incoherent text (see evals/RESULTS_LOG.md, 2026-07-10 GPU run).

This module instead asks a real LLM teacher to *write* a coherent i+1 story for each scenario, runs
the validator-guided rewrite loop, then applies a **coherence/quality judge gate** so only stories
that are both spec-passing *and* actually good are kept. It reuses the compact-KNOWN convention
(known = the story's own content words minus targets) so records match dataset_v1's schema and the
eval's curated setup, and coverage passes by construction.

    # Pilot (needs OPENAI_API_KEY + a capable teacher/judge model in .env):
    python -m islm.datagen.teacher --n 1000 --language en --out data/generated/teacher_en

    # Offline smoke (deterministic MockLLM, no key/network):
    python -m islm.datagen.teacher --n 5 --language en --mock --out /tmp/teacher_en
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

from ..config import DEFAULT_THRESHOLDS, Thresholds
from ..llm.client import LLMClient, get_client
from ..validators import validate_story
from ..vocab.lemmatize import Lemmatizer, get_analyzer
from ..vocab.wordlists import VOCAB_DIR, Vocabulary
from .generate import Example, generate_story, rewrite_story
from .scenarios import Scenario
from .synth import TARGET_POOLS, _compact_known


@dataclass
class TeacherConfig:
    """Quality gate for kept stories (on top of the deterministic hard-pass)."""

    max_rewrites: int = 4  # validator-guided fix attempts before giving up on a scenario
    temperature: float = 0.8  # teacher sampling; some variety so stories aren't near-duplicates
    # Judge gate — the point of this module. A story must be coherent AND have some spark, not just
    # tick the boxes. Anchored 0/1/2 per llm.prompts._RUBRIC.
    min_coherence: int = 2
    min_task_quality: int = 2
    min_interestingness: int = 1


def _theme_pool(language: str) -> list[str]:
    from ..vocab.languages import get_language

    return list(get_language(language).themes) or ["a small adventure"]


def _sample_targets(rng: random.Random, pools: dict[str, list[str]]) -> list[tuple[str, str]]:
    """2-3 distinct POS-typed targets (bias to 2), same shape as synth."""
    typed = [(pos, w) for pos, words in pools.items() for w in words]
    k = rng.choice([2, 2, 3])
    while True:
        picks = rng.sample(typed, k)
        if len({w.lower() for _, w in picks}) == k:
            return picks


def _judge_gate(judge: dict | None, config: TeacherConfig) -> bool:
    """True if the story clears the coherence/quality bar (drops flat, gamed text)."""
    if judge is None:
        return True  # no judge available -> deterministic-only (still better prose than templates)
    return (
        judge.get("coherence", 0) >= config.min_coherence
        and judge.get("task_quality", 0) >= config.min_task_quality
        and judge.get("interestingness", 0) >= config.min_interestingness
    )


def make_teacher_example(
    language: str,
    targets: list[str],
    theme: str,
    idx: int,
    client: LLMClient,
    analyzer: Lemmatizer,
    config: TeacherConfig,
    judge_client: LLMClient | None,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> Example:
    """Teacher-write one story, fix it to hard-pass, judge-gate it. Returns an Example (kept flag).

    The prompt hands the LLM the language's baseline vocab as the allowed palette (small: ~66-146
    words) plus the targets; the record's KNOWN_WORDS is then scoped to what the story actually
    used (compact-known), matching dataset_v1 and the eval.
    """
    palette = sorted(Vocabulary.from_csv(VOCAB_DIR / language / "baseline.csv").lemmas)
    target_set = {t.lower() for t in targets}

    # Generation scenario: give the model the baseline palette to write within.
    gen_scenario = Scenario(
        id=f"{language}-teacher-{idx:06d}",
        language=language,
        level="baseline",
        theme=theme,
        target_words=targets,
        known=palette,
    )
    story = generate_story(gen_scenario, client, thresholds, config.temperature)

    # Scope known to the story's own words (minus targets) so coverage passes by construction,
    # then run the validator-guided rewrite loop against that scoped set.
    def scoped(text: str) -> tuple[Scenario, set[str]]:
        known = _compact_known(text, target_set, analyzer)
        return (
            Scenario(gen_scenario.id, language, "baseline", theme, targets, known),
            set(known),
        )

    scenario, known = scoped(story)
    report = validate_story(story, known, target_set, analyzer, thresholds, language=language)
    passes = 0
    while not report.hard_pass and passes < config.max_rewrites:
        story = rewrite_story(scenario, story, report.failures(), client, thresholds)
        scenario, known = scoped(story)
        report = validate_story(story, known, target_set, analyzer, thresholds, language=language)
        passes += 1

    judge = None
    if judge_client is not None:
        from ..eval.judge import judge_story

        judge = judge_story(scenario, story, judge_client)

    kept = report.hard_pass and _judge_gate(judge, config)
    return Example(scenario, story, report, passes, judge, kept)


def generate(
    language: str,
    n: int,
    out_dir: Path,
    *,
    client: LLMClient,
    judge_client: LLMClient | None,
    config: TeacherConfig | None = None,
    seed: int = 0,
    max_attempts: int | None = None,
) -> dict:
    """Generate up to `n` kept teacher stories; write train/val/test (80/10/10) + stats."""
    config = config or TeacherConfig()
    analyzer = get_analyzer(language)
    pools = TARGET_POOLS[language]
    themes = _theme_pool(language)
    rng = random.Random(seed)
    max_attempts = max_attempts or n * 4

    records: list[dict] = []
    kept = attempts = rejected = 0
    while kept < n and attempts < max_attempts:
        attempts += 1
        targets = [w for _, w in _sample_targets(rng, pools)]
        theme = rng.choice(themes)
        ex = make_teacher_example(
            language, targets, theme, kept, client, analyzer, config, judge_client
        )
        if not ex.kept:
            rejected += 1
            continue
        rec = ex.to_record()
        rec["metadata"]["source"] = "teacher-v2"
        records.append(rec)
        kept += 1

    rng.shuffle(records)
    n_tr, n_va = int(kept * 0.8), int(kept * 0.1)
    splits = {
        "train": records[:n_tr],
        "val": records[n_tr : n_tr + n_va],
        "test": records[n_tr + n_va :],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, rs in splits.items():
        with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {
        "language": language,
        "requested": n,
        "kept": kept,
        "attempts": attempts,
        "rejected": rejected,
        "keep_rate": round(kept / attempts, 4) if attempts else 0.0,
        "splits": {k: len(v) for k, v in splits.items()},
        "source": "teacher-v2",
    }
    with open(out_dir / "teacher_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description="LLM teacher-regenerated i+1 stories (quality path).")
    p.add_argument("--n", type=int, default=1000, help="Target number of kept stories.")
    p.add_argument("--language", default="en")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--teacher-model", default=None, help="Teacher model (default: from env).")
    p.add_argument("--judge-model", default=None, help="Judge model (default: JUDGE_MODEL env).")
    p.add_argument("--no-judge", action="store_true", help="Skip the judge gate (deterministic).")
    p.add_argument("--mock", action="store_true", help="Offline MockLLM for teacher + judge.")
    p.add_argument("--max-attempts", type=int, default=None)
    args = p.parse_args()

    if args.mock:
        client = judge_client = get_client(mock=True)
    else:
        client = get_client(args.teacher_model)
        judge_client = None if args.no_judge else get_client(args.judge_model)

    stats = generate(
        args.language, args.n, args.out,
        client=client, judge_client=judge_client, seed=args.seed,
        max_attempts=args.max_attempts,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
