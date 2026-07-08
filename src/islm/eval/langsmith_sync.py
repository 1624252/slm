"""Push the golden set and eval runs to LangSmith (augment mode).

LangSmith is the eval-tracking UI layer; it does NOT replace the deterministic validators or the
committed `runs.jsonl`/leaderboard — those stay the source of truth. This module only *uploads*:

- `golden.jsonl` -> a LangSmith **dataset** (inputs = the scenario, outputs = the reference story
  + expected metrics), so the golden set is visible/versioned in the LangSmith UI.
- an eval `results_<lang>.json` -> a LangSmith **experiment** (one run per scenario, with the
  deterministic scores attached), so runs can be compared across the CPU->GPU jump.

Everything is gated on `LANGSMITH_API_KEY`: with no key it is a no-op that prints how to enable it,
so the offline test suite never depends on the network. `langsmith` is an optional dependency
(`pip install -e .[langsmith]`) and is imported lazily.

    export LANGSMITH_API_KEY=lsv2_...        # required to actually push
    python -m islm.eval.langsmith_sync golden   --golden evals/golden/golden.jsonl
    python -m islm.eval.langsmith_sync results  --results evals/day3_v3/results_en.json \
        --experiment day3-seed-lora-v3
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

DATASET_PREFIX = "islm"  # dataset names: "islm-golden", "islm-heldout-en", ...


def _client():
    """Return a LangSmith Client, or None if no key / package (augment mode is optional)."""
    if not os.getenv("LANGSMITH_API_KEY"):
        return None
    try:
        from langsmith import Client
    except ImportError:
        print("langsmith not installed; run `pip install -e .[langsmith]`")
        return None
    return Client()


def golden_to_examples(golden_path: Path) -> list[dict]:
    """Convert golden.jsonl records to LangSmith {inputs, outputs, metadata} examples.

    inputs = the model's task (scenario fields + the rendered prompt). outputs = the reference
    story and the expected deterministic metrics (the "correct" the golden set defines).
    """
    examples = []
    for line in golden_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        msgs = {m["role"]: m["content"] for m in rec["messages"]}
        meta = rec["metadata"]
        examples.append(
            {
                "inputs": {
                    "language": rec["language"],
                    "target_words": rec["target_words"],
                    "system": msgs.get("system", ""),
                    "user": msgs.get("user", ""),
                },
                "outputs": {
                    "reference_story": msgs.get("assistant", ""),
                    "hard_pass": meta.get("hard_pass"),
                    "oov_rate": meta.get("oov_rate"),
                    "coverage": meta.get("coverage"),
                },
                "metadata": {
                    "id": rec["id"],
                    "tone": meta.get("tone"),
                    "keywords": meta.get("keywords"),
                    "target_tier": meta.get("target_tier"),
                    "category": meta.get("category"),
                    "difficulty": meta.get("difficulty"),
                    "source": meta.get("source"),
                },
            }
        )
    return examples


def push_golden(golden_path: Path, dataset_name: str = f"{DATASET_PREFIX}-golden") -> int:
    """Upload the golden set as a LangSmith dataset. Returns the number of examples pushed."""
    examples = golden_to_examples(golden_path)
    client = _client()
    if client is None:
        print(
            f"[dry-run] {len(examples)} golden examples ready for dataset '{dataset_name}'.\n"
            "Set LANGSMITH_API_KEY (and pip install -e .[langsmith]) to push."
        )
        return 0
    if client.has_dataset(dataset_name=dataset_name):
        ds = client.read_dataset(dataset_name=dataset_name)
    else:
        ds = client.create_dataset(dataset_name, description="i+1 story golden set (Layer 1)")
    client.create_examples(
        inputs=[e["inputs"] for e in examples],
        outputs=[e["outputs"] for e in examples],
        metadata=[e["metadata"] for e in examples],
        dataset_id=ds.id,
    )
    print(f"pushed {len(examples)} examples -> LangSmith dataset '{dataset_name}'")
    return len(examples)


def results_to_records(results_path: Path) -> list[dict]:
    """Flatten an eval results JSON into per-scenario score records (tuned side if present)."""
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    rows = payload.get("tuned_rows") or payload.get("base_rows") or []
    return [
        {
            "id": r["id"],
            "story": r.get("story", ""),
            "hard_pass": r.get("hard_pass"),
            "oov_rate": r.get("oov_rate"),
            "failures": r.get("failures", []),
        }
        for r in rows
    ]


def push_results(results_path: Path, experiment: str) -> int:
    """Log a finished eval run's per-scenario scores to LangSmith as an experiment."""
    records = results_to_records(results_path)
    client = _client()
    if client is None:
        print(
            f"[dry-run] {len(records)} scored rows ready for experiment '{experiment}'.\n"
            "Set LANGSMITH_API_KEY (and pip install -e .[langsmith]) to push."
        )
        return 0
    # Log each scored row as a run under the experiment project; scores are the deterministic
    # metrics we already computed offline (no re-generation, no judge needed).
    for r in records:
        client.create_run(
            name=f"eval:{r['id']}",
            run_type="chain",
            project_name=experiment,
            inputs={"scenario_id": r["id"]},
            outputs={"story": r["story"]},
            extra={"metadata": {"hard_pass": r["hard_pass"], "oov_rate": r["oov_rate"],
                                "failures": r["failures"]}},
        )
    print(f"logged {len(records)} rows -> LangSmith experiment '{experiment}'")
    return len(records)


def main() -> None:
    p = argparse.ArgumentParser(description="Push golden set / eval runs to LangSmith (augment).")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("golden", help="Upload the golden set as a dataset.")
    g.add_argument("--golden", type=Path, default=Path("evals/golden/golden.jsonl"))
    g.add_argument("--dataset-name", default=f"{DATASET_PREFIX}-golden")

    r = sub.add_parser("results", help="Log an eval results JSON as an experiment.")
    r.add_argument("--results", type=Path, required=True)
    r.add_argument("--experiment", required=True, help="Experiment / project name in LangSmith.")

    args = p.parse_args()
    if args.cmd == "golden":
        push_golden(args.golden, args.dataset_name)
    else:
        push_results(args.results, args.experiment)


if __name__ == "__main__":
    main()
