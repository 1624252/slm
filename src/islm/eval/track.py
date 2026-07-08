"""Results tracking over time.

Every base-vs-tuned eval appends one JSON line to `evals/runs.jsonl` (an append-only history,
git-tracked) and regenerates `evals/LEADERBOARD.md` (a compact base-vs-tuned table, newest first),
so progress across runs and days is visible in one place. This is the lightweight "CSV/JSONL"
tracking option from PRD 11 (Weights & Biases is the optional heavier alternative) and the
per-run eval record of PRD 13.4.

Wire it into the harness with `python -m islm.eval.run ... --track --run-label <name>`, or record
a finished results file / rebuild the board from history here:

    # Record a results file and refresh the leaderboard:
    python -m islm.eval.track --results evals/day3/results_en.json --run-label day3-seed-lora \
        --dataset data/curated/seed --epochs 3

    # Rebuild the leaderboard from the existing history (no new run):
    python -m islm.eval.track --rebuild
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..config import EVALS_DIR

RUNS_PATH = EVALS_DIR / "runs.jsonl"
LEADERBOARD_PATH = EVALS_DIR / "LEADERBOARD.md"

# Metrics tracked across runs: (results key, column label, higher-is-better).
METRICS = [
    ("hard_pass_rate", "Hard-pass", True),
    ("mean_oov_rate", "OOV", False),
    ("one_new_word_pass_rate", "<=1-new", True),
    ("recurrence_pass_rate", "Recurrence", True),
    ("judge_overall", "Judge", True),
    ("mean_inferability", "Inferability", True),
]
_METRIC_KEYS = [key for key, _, _ in METRICS]


@dataclass
class RunRecord:
    """One eval run's headline numbers plus the context needed to reproduce/compare it."""

    timestamp: str
    label: str
    language: str
    commit: str | None = None
    base_model: str | None = None
    tuned_model: str | None = None
    tuned_adapter: str | None = None
    dataset: str | None = None
    train_examples: int | None = None
    n_scenarios: int | None = None
    epochs: float | None = None
    max_steps: int | None = None
    base: dict = field(default_factory=dict)  # metric key -> value
    tuned: dict = field(default_factory=dict)
    adversarial: dict = field(default_factory=dict)  # {"base": hard_pass, "tuned": hard_pass}
    notes: str | None = None


def git_commit(short: bool = True) -> str | None:
    """Short git SHA of HEAD, or None if unavailable (not a repo / git missing)."""
    cmd = ["git", "rev-parse", *(["--short"] if short else []), "HEAD"]
    try:
        r = subprocess.run(cmd, cwd=EVALS_DIR.parent, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    out = r.stdout.strip()
    return out if r.returncode == 0 and out else None


def _count_train_examples(dataset: str | None) -> int | None:
    """Number of training records in `<dataset>/train.jsonl`, if that file exists."""
    if not dataset:
        return None
    path = Path(dataset) / "train.jsonl"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _pick(metrics: dict) -> dict:
    """Keep only the tracked metrics that are present (rounded floats)."""
    out = {}
    for key in _METRIC_KEYS:
        if key in metrics and metrics[key] is not None:
            out[key] = round(float(metrics[key]), 4)
    return out


def record_from_results(
    results: dict,
    *,
    label: str,
    language: str,
    dataset: str | None = None,
    base_model: str | None = None,
    tuned_model: str | None = None,
    tuned_adapter: str | None = None,
    epochs: float | None = None,
    max_steps: int | None = None,
    notes: str | None = None,
    commit: str | None = None,
    timestamp: str | None = None,
) -> RunRecord:
    """Build a `RunRecord` from an eval results payload (the dict written by `islm.eval.run`)."""
    base = results.get("base") or {}
    tuned = results.get("tuned") or {}
    adv = results.get("adversarial") or {}
    adv_metrics: dict = {}
    for side in ("base", "tuned"):
        side_metrics = adv.get(side) or {}
        if side_metrics.get("hard_pass_rate") is not None:
            adv_metrics[side] = round(float(side_metrics["hard_pass_rate"]), 4)

    return RunRecord(
        timestamp=timestamp or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        label=label,
        language=language,
        commit=commit if commit is not None else git_commit(),
        base_model=base_model or base.get("model"),
        tuned_model=tuned_model or tuned.get("model"),
        tuned_adapter=tuned_adapter,
        dataset=dataset,
        train_examples=_count_train_examples(dataset),
        n_scenarios=base.get("n") or tuned.get("n"),
        epochs=epochs,
        max_steps=max_steps,
        base=_pick(base),
        tuned=_pick(tuned),
        adversarial=adv_metrics,
        notes=notes,
    )


def append_run(record: RunRecord, runs_path: Path = RUNS_PATH) -> None:
    """Append one run to the JSONL history (created if missing)."""
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    with open(runs_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def load_runs(runs_path: Path = RUNS_PATH) -> list[dict]:
    """Read the run history (empty list if none yet)."""
    if not runs_path.exists():
        return []
    with open(runs_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _fmt(value) -> str:
    if isinstance(value, bool):  # guard: bool is an int subclass
        return str(value)
    if isinstance(value, float):
        return f"{value:.3f}"
    return "-" if value is None else str(value)


def _cell(base_value, tuned_value) -> str:
    """`base->tuned (delta)` when both exist, else whichever is present, else `-`."""
    if base_value is None and tuned_value is None:
        return "-"
    if tuned_value is None:
        return _fmt(base_value)
    if base_value is None:
        return _fmt(tuned_value)
    return f"{_fmt(base_value)}->{_fmt(tuned_value)} ({tuned_value - base_value:+.3f})"


_LEGEND = (
    "\n_Cells are `base->tuned (delta)`; delta = tuned - base. Higher is better for every column "
    "except **OOV** (lower is better), so a negative OOV delta is an improvement. Rates are in "
    "[0,1]; Judge is the mean rubric score in [0,2]. Adv = hard-pass on the adversarial set "
    "(robustness). Blank/`-` = not measured that run (e.g. no judge without an API key)._\n"
)


def leaderboard_markdown(runs: list[dict]) -> str:
    """Render the run history as one base-vs-tuned table, newest first."""
    header = (
        "# Results leaderboard\n\n"
        "Base-vs-tuned numbers per eval run, newest first. Regenerated by `islm.eval.track` from "
        "`evals/runs.jsonl` (the append-only history); do not edit by hand."
    )
    if not runs:
        return header + "\n\n_No runs recorded yet._\n"

    cols = [
        "When",
        "Label",
        "Lang",
        "n",
        "Hard-pass",
        "OOV",
        "<=1-new",
        "Recurrence",
        "Judge",
        "Adv",
        "Data",
        "Commit",
    ]
    lines = [header, "", "| " + " | ".join(cols) + " |", "|" + " --- |" * len(cols)]
    for r in sorted(runs, key=lambda x: x.get("timestamp", ""), reverse=True):
        base, tuned = r.get("base") or {}, r.get("tuned") or {}
        adv = r.get("adversarial") or {}
        data = r.get("dataset") or "-"
        n_train = r.get("train_examples")
        data_cell = data + (f" (n={n_train})" if n_train is not None else "")
        row = [
            (r.get("timestamp") or "")[:10],
            r.get("label") or "-",
            r.get("language") or "-",
            _fmt(r.get("n_scenarios")),
            _cell(base.get("hard_pass_rate"), tuned.get("hard_pass_rate")),
            _cell(base.get("mean_oov_rate"), tuned.get("mean_oov_rate")),
            _cell(base.get("one_new_word_pass_rate"), tuned.get("one_new_word_pass_rate")),
            _cell(base.get("recurrence_pass_rate"), tuned.get("recurrence_pass_rate")),
            _cell(base.get("judge_overall"), tuned.get("judge_overall")),
            _cell(adv.get("base"), adv.get("tuned")),
            data_cell,
            r.get("commit") or "-",
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n" + _LEGEND


def write_leaderboard(runs_path: Path = RUNS_PATH, out_path: Path = LEADERBOARD_PATH) -> Path:
    """(Re)generate the leaderboard markdown from the run history."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(leaderboard_markdown(load_runs(runs_path)), encoding="utf-8")
    return out_path


def track_results(
    results_path: str | Path,
    *,
    label: str,
    language: str | None = None,
    dataset: str | None = None,
    base_model: str | None = None,
    tuned_model: str | None = None,
    tuned_adapter: str | None = None,
    epochs: float | None = None,
    max_steps: int | None = None,
    notes: str | None = None,
    timestamp: str | None = None,
    runs_path: Path = RUNS_PATH,
    leaderboard_path: Path = LEADERBOARD_PATH,
) -> RunRecord:
    """Record an eval results file and refresh the leaderboard. Returns the appended record."""
    results_path = Path(results_path)
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    if language is None:  # infer from results_<lang>.json
        stem = results_path.stem
        language = stem.split("_")[-1] if "_" in stem else "?"
    record = record_from_results(
        payload,
        label=label,
        language=language,
        dataset=dataset,
        base_model=base_model,
        tuned_model=tuned_model,
        tuned_adapter=tuned_adapter,
        epochs=epochs,
        max_steps=max_steps,
        notes=notes,
        timestamp=timestamp,
    )
    append_run(record, runs_path)
    write_leaderboard(runs_path, leaderboard_path)
    return record


def main() -> None:
    p = argparse.ArgumentParser(
        description="Track eval results over time (runs log + leaderboard)."
    )
    p.add_argument("--results", type=Path, default=None, help="Eval results_<lang>.json to record.")
    p.add_argument(
        "--run-label", default=None, help="Short name for this run (e.g. day3-seed-lora)."
    )
    p.add_argument("--language", default=None, help="Override; else inferred from the filename.")
    p.add_argument(
        "--dataset", default=None, help="Training dataset dir (for train-example count)."
    )
    p.add_argument("--base-model", default=None)
    p.add_argument("--tuned-model", default=None)
    p.add_argument("--tuned-adapter", default=None)
    p.add_argument("--epochs", type=float, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--notes", default=None)
    p.add_argument(
        "--timestamp", default=None, help="Override run time (ISO 8601); for backfilling."
    )
    p.add_argument("--runs", type=Path, default=RUNS_PATH, help="Runs-history JSONL.")
    p.add_argument(
        "--leaderboard", type=Path, default=LEADERBOARD_PATH, help="Leaderboard markdown."
    )
    p.add_argument(
        "--rebuild", action="store_true", help="Only rebuild the leaderboard from the runs log."
    )
    args = p.parse_args()

    if args.rebuild and not args.results:
        out = write_leaderboard(args.runs, args.leaderboard)
        print(f"rebuilt {out} from {args.runs} ({len(load_runs(args.runs))} runs)")
        return
    if not args.results or not args.run_label:
        p.error("provide --results and --run-label (or --rebuild)")

    record = track_results(
        args.results,
        label=args.run_label,
        language=args.language,
        dataset=args.dataset,
        base_model=args.base_model,
        tuned_model=args.tuned_model,
        tuned_adapter=args.tuned_adapter,
        epochs=args.epochs,
        max_steps=args.max_steps,
        notes=args.notes,
        timestamp=args.timestamp,
        runs_path=args.runs,
        leaderboard_path=args.leaderboard,
    )
    print(
        f"recorded '{record.label}' ({record.language}) -> {args.runs}; "
        f"leaderboard -> {args.leaderboard}"
    )


if __name__ == "__main__":
    main()
