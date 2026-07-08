"""Replay harness (Layer 4) — record once, score anytime.

An eval run stores every generated story in `results_<lang>.json` (`base_rows`/`tuned_rows`).
Generating them costs tokens/time; **re-scoring them is free**. This module reads those frozen
stories back through the deterministic validators (no model load, no network) and re-emits the
metrics — so you can:

  * regression-check that a scoring change didn't move old numbers,
  * re-score a frozen snapshot after a validator/threshold change,
  * apply later human-annotated ground truth to a recorded run.

The stories' scenarios (KNOWN_WORDS / TARGET_WORDS) are matched by `id` from the committed
held-out scenario files, so no scenario data needs to live in the results JSON.

    python -m islm.eval.replay --results evals/day3_v3/results_en.json --language en
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import EVALS_DIR
from ..datagen.scenarios import load_scenarios
from ..validators import validate_story
from ..vocab.lemmatize import get_analyzer


def _scenario_index(language: str, curated: bool, scenarios_path: Path | None) -> dict:
    """id -> Scenario, from the committed held-out file used for this language."""
    if scenarios_path is None:
        name = f"heldout_small_{language}.jsonl" if curated else f"heldout_{language}.jsonl"
        scenarios_path = EVALS_DIR / "scenarios" / name
    if not scenarios_path.exists():
        raise SystemExit(f"scenario file not found: {scenarios_path}")
    return {s.id: s for s in load_scenarios(scenarios_path)}


def replay_rows(rows: list[dict], scenarios: dict, language: str) -> list[dict]:
    """Re-score each stored story through the validators. Returns per-row re-computed metrics."""
    lem = get_analyzer(language)
    out = []
    for r in rows:
        scn = scenarios.get(r["id"])
        if scn is None:
            continue  # story with no matching scenario (e.g. sampled ad-hoc) — skip
        report = validate_story(r["story"], scn.known_set(), scn.target_set(), lem)
        out.append(
            {
                "id": r["id"],
                "hard_pass": report.hard_pass,
                "oov_rate": round(report.coverage.oov_rate, 4),
                "recorded_hard_pass": r.get("hard_pass"),
                "recorded_oov_rate": r.get("oov_rate"),
            }
        )
    return out


def summarize(scored: list[dict]) -> dict:
    n = len(scored)
    if not n:
        return {"n": 0}
    return {
        "n": n,
        "hard_pass_rate": round(sum(s["hard_pass"] for s in scored) / n, 4),
        "mean_oov_rate": round(sum(s["oov_rate"] for s in scored) / n, 4),
        "matches_recorded": all(
            s["recorded_hard_pass"] is None or s["hard_pass"] == s["recorded_hard_pass"]
            for s in scored
        ),
    }


def replay(results_path: Path, language: str, side: str, curated: bool,
           scenarios_path: Path | None = None) -> dict:
    """Re-score a stored run. side = 'tuned' or 'base'. Returns {base?, tuned?} summaries."""
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    scenarios = _scenario_index(language, curated, scenarios_path)
    result = {}
    for which in (["base", "tuned"] if side == "both" else [side]):
        rows = payload.get(f"{which}_rows")
        if rows:
            result[which] = summarize(replay_rows(rows, scenarios, language))
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Replay (re-score) stored eval stories, no model.")
    p.add_argument("--results", type=Path, required=True, help="results_<lang>.json to replay.")
    p.add_argument("--language", default=None, help="Override; else inferred from the filename.")
    p.add_argument("--side", default="both", choices=["base", "tuned", "both"])
    p.add_argument("--curated", action="store_true", help="Match against the small curated set.")
    p.add_argument("--scenarios", type=Path, default=None, help="Explicit held-out JSONL.")
    args = p.parse_args()

    lang = args.language or args.results.stem.split("_")[-1]
    # Default to the curated held-out set (what the tracked runs used).
    curated = args.curated or args.scenarios is None
    result = replay(args.results, lang, args.side, curated=curated, scenarios_path=args.scenarios)
    for which, summ in result.items():
        print(f"[{which}] {summ}")
    if any(s.get("matches_recorded") is False for s in result.values()):
        print("WARNING: replayed scores differ from recorded — a validator or the data changed.")


if __name__ == "__main__":
    main()
