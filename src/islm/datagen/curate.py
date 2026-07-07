"""Second pass: review a generated dataset, decide good vs bad, and filter out the bad.

The generation pipeline already applies the hard validators + a per-item judge (first pass).
This second pass runs over the *whole corpus* to catch what per-item checks miss:

- **Duplicates** — exact and structural near-duplicates (teachers repeat themselves at scale).
- **Degenerate repetition** — low unique-sentence ratio or type-token ratio (filler, not a story).
- **Too short** — fewer than a minimum number of sentences.
- **Re-validation** — re-run the deterministic backbone as defense-in-depth.
- **Stricter re-judge** — optional LLM judge with a higher bar than generation.

Kept records go to ``<out>/{train,val,test}.jsonl``; rejects (annotated with reasons) go to
``<out>/rejects.jsonl``; a human-readable ``curation_report.md`` summarizes the decisions.

    python -m islm.datagen.curate --in data/generated/en --out data/curated/en
    python -m islm.datagen.curate --in data/generated/zh --out data/curated/zh --judge-model gpt-5
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ..llm.client import LLMClient, get_client
from ..validators import validate_story
from ..vocab.lemmatize import Lemmatizer, get_analyzer
from ..vocab.tokenize import split_sentences

_SPLITS = ("train", "val", "test")


@dataclass
class CurationConfig:
    min_sentences: int = 4
    min_unique_sentence_ratio: float = 0.6  # reject "X is big. X is big. X is big."
    min_type_token_ratio: float = 0.30  # lexical variety of content
    revalidate: bool = True  # re-run the hard validators
    judge_min_spec_adherence: int = 2
    judge_min_mean: float = 1.5


@dataclass
class CurationResult:
    kept: dict[str, list[dict]] = field(default_factory=lambda: {s: [] for s in _SPLITS})
    rejects: list[dict] = field(default_factory=list)
    reasons: Counter = field(default_factory=Counter)

    @property
    def n_kept(self) -> int:
        return sum(len(v) for v in self.kept.values())


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()


def _known_target(record: dict) -> tuple[set[str], set[str]]:
    user = next((m["content"] for m in record.get("messages", []) if m["role"] == "user"), "")
    m = re.search(r"KNOWN_WORDS:\s*(.+)", user)
    known = {w.strip().lower() for w in m.group(1).split(",")} if m else set()
    target = {w.strip().lower() for w in record.get("target_words", [])}
    return known, target


def _story_of(record: dict) -> str:
    return next((m["content"] for m in record.get("messages", []) if m["role"] == "assistant"), "")


def _quality_reasons(record: dict, lemmatizer: Lemmatizer, config: CurationConfig) -> list[str]:
    """Deterministic good/bad checks (no LLM). Returns a list of failure reasons (empty = good)."""
    story = _story_of(record)
    sentences = split_sentences(story)
    reasons: list[str] = []

    if len(sentences) < config.min_sentences:
        reasons.append("too_short")

    if sentences:
        uniq_ratio = len(set(_normalized(s) for s in sentences)) / len(sentences)
        if uniq_ratio < config.min_unique_sentence_ratio:
            reasons.append("repetitive_sentences")

    known, target = _known_target(record)
    tokens = [t.lemma for s in sentences for t in lemmatizer.analyze(s) if t.is_word]
    if tokens:
        ttr = len(set(tokens)) / len(tokens)
        if ttr < config.min_type_token_ratio:
            reasons.append("low_lexical_variety")

    if config.revalidate and known:
        report = validate_story(
            story, known, target, lemmatizer, language=record.get("language", "en")
        )
        if not report.hard_pass:
            reasons.append("failed_revalidation")

    return reasons


def curate(
    records_by_split: dict[str, list[dict]],
    config: CurationConfig | None = None,
    judge_client: LLMClient | None = None,
) -> CurationResult:
    """Run the second pass over a whole dataset and return kept/rejected + reason stats."""
    config = config or CurationConfig()
    result = CurationResult()
    analyzers: dict[str, Lemmatizer] = {}
    seen_exact: set[str] = set()
    seen_signature: set[tuple] = set()

    for split in _SPLITS:
        for record in records_by_split.get(split, []):
            lang = record.get("language", "en")
            lemmatizer = analyzers.setdefault(lang, get_analyzer(lang))
            story = _story_of(record)
            reasons: list[str] = []

            # Corpus-level dedup.
            norm = _normalized(story)
            signature = (lang, tuple(sorted(record.get("target_words", []))), norm[:60])
            if norm in seen_exact:
                reasons.append("duplicate")
            elif signature in seen_signature:
                reasons.append("near_duplicate")
            else:
                seen_exact.add(norm)
                seen_signature.add(signature)

            # Per-item quality (skip if already a dup).
            if not reasons:
                reasons = _quality_reasons(record, lemmatizer, config)

            # Optional stricter LLM re-judge.
            if not reasons and judge_client is not None:
                reasons = _judge_reasons(record, judge_client, config)

            if reasons:
                result.reasons.update(reasons)
                annotated = {**record, "reject_reasons": reasons}
                result.rejects.append(annotated)
            else:
                result.kept[split].append(record)

    return result


def _judge_reasons(record: dict, client: LLMClient, config: CurationConfig) -> list[str]:
    from ..datagen.scenarios import Scenario
    from ..eval.judge import judge_story

    scenario = Scenario(
        id=record.get("id", ""),
        language=record.get("language", "en"),
        level=record.get("level", ""),
        theme=record.get("theme", ""),
        target_words=record.get("target_words", []),
        known=[],
    )
    scores = judge_story(scenario, _story_of(record), client)
    mean = sum(v for v in scores.values() if isinstance(v, int)) / max(
        1, sum(1 for v in scores.values() if isinstance(v, int))
    )
    reasons = []
    if scores.get("spec_adherence", 0) < config.judge_min_spec_adherence:
        reasons.append("judge_spec_adherence")
    if mean < config.judge_min_mean:
        reasons.append("judge_low_quality")
    return reasons


# --- file I/O + CLI -------------------------------------------------------------------------


def _read_splits(in_dir: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for split in _SPLITS:
        path = in_dir / f"{split}.jsonl"
        out[split] = (
            [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
            if path.exists()
            else []
        )
    return out


def _write_results(out_dir: Path, result: CurationResult, config: CurationConfig) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, records in result.kept.items():
        with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out_dir / "rejects.jsonl", "w", encoding="utf-8") as f:
        for r in result.rejects:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = result.n_kept + len(result.rejects)
    stats = {
        "input_records": total,
        "kept": result.n_kept,
        "rejected": len(result.rejects),
        "keep_rate": round(result.n_kept / total, 4) if total else 0.0,
        "reject_reasons": dict(result.reasons.most_common()),
        "kept_by_split": {s: len(v) for s, v in result.kept.items()},
    }
    with open(out_dir / "curation_report.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    _write_report_md(out_dir / "curation_report.md", stats, config)
    return stats


def _write_report_md(path: Path, stats: dict, config: CurationConfig) -> None:
    lines = [
        "# Curation report (second pass)",
        "",
        f"- Input records: **{stats['input_records']}**",
        f"- Kept: **{stats['kept']}** | Rejected: **{stats['rejected']}** "
        f"| Keep rate: **{stats['keep_rate']:.1%}**",
        f"- Kept by split: {stats['kept_by_split']}",
        "",
        "## Rejection reasons",
        "",
        "| Reason | Count |",
        "| --- | --- |",
    ]
    lines += [f"| {reason} | {count} |" for reason, count in stats["reject_reasons"].items()]
    lines += ["", "## Thresholds", "", "```", json.dumps(config.__dict__, indent=2), "```", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Second-pass curation for a generated dataset.")
    p.add_argument("--in", dest="in_dir", type=Path, required=True)
    p.add_argument("--out", dest="out_dir", type=Path, required=True)
    p.add_argument("--judge-model", default=None, help="Optional stricter LLM re-judge.")
    p.add_argument("--mock", action="store_true", help="Use the offline MockLLM for judging.")
    args = p.parse_args()

    judge_client = None
    if args.mock:
        judge_client = get_client(mock=True)
    elif args.judge_model:
        judge_client = get_client(args.judge_model)

    result = curate(_read_splits(args.in_dir), judge_client=judge_client)
    stats = _write_results(args.out_dir, result, CurationConfig())
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
