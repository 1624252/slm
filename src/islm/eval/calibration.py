"""Judge calibration (Layer 5) — trust the LLM judge only after it agrees with humans.

A judge with a bad rubric produces confident, wrong scores. Before using the LLM-as-judge rubric
(`llm/prompts.py`), score ~20 stories by hand, run the judge on the same stories, and check the
per-dimension correlation. If any dimension correlates < 0.8 with human scores, the rubric anchors
for that dimension are ambiguous — fix them before trusting the judge (PDF Layer 5).

This module computes the correlation from a paired-scores file; it does NOT call the judge itself
(that needs an API key — do it on Colab, then feed the scores here). Offline and dependency-free.

    # scores.json: {"human": [{dim: score,...}, ...], "judge": [{dim: score,...}, ...]}
    python -m islm.eval.calibration --scores evals/calibration/scores.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..llm.prompts import JUDGE_DIMENSIONS

MIN_CORRELATION = 0.8  # PDF threshold: below this, the rubric is broken, not the model.


def pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation; None if undefined (n<2 or a constant vector)."""
    n = len(xs)
    if n < 2 or len(ys) != n:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    num = sum(a * b for a, b in zip(dx, dy, strict=True))
    den = (sum(a * a for a in dx) * sum(b * b for b in dy)) ** 0.5
    if den == 0:
        return None  # one side is constant -> correlation undefined
    return num / den


def calibrate(human: list[dict], judge: list[dict]) -> dict:
    """Per-dimension human-vs-judge correlation + a pass flag (>= 0.8, or exact agreement)."""
    if len(human) != len(judge):
        raise ValueError(f"human ({len(human)}) and judge ({len(judge)}) counts differ")
    report: dict = {"n": len(human), "dimensions": {}, "ok": True}
    for dim in JUDGE_DIMENSIONS:
        h = [float(row.get(dim, 0)) for row in human]
        j = [float(row.get(dim, 0)) for row in judge]
        r = pearson(h, j)
        exact = sum(a == b for a, b in zip(h, j, strict=True)) / len(h) if h else 0.0
        # A constant-but-identical pair (e.g. all 2s) is perfect agreement even though r is None.
        passed = (r is not None and r >= MIN_CORRELATION) or (r is None and exact == 1.0)
        report["dimensions"][dim] = {
            "correlation": None if r is None else round(r, 3),
            "exact_agreement": round(exact, 3),
            "passed": passed,
        }
        report["ok"] = report["ok"] and passed
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Calibrate the LLM judge against human scores.")
    p.add_argument("--scores", type=Path, required=True, help="JSON: 'human' and 'judge' lists.")
    args = p.parse_args()

    data = json.loads(args.scores.read_text(encoding="utf-8"))
    report = calibrate(data["human"], data["judge"])
    print(f"calibration on {report['n']} paired scores (threshold r >= {MIN_CORRELATION}):")
    for dim, d in report["dimensions"].items():
        flag = "ok" if d["passed"] else "LOW — fix this rubric anchor"
        print(f"  {dim:26} r={d['correlation']} exact={d['exact_agreement']}  [{flag}]")
    print("JUDGE TRUSTWORTHY" if report["ok"] else "JUDGE NOT CALIBRATED — do not trust its scores")


if __name__ == "__main__":
    main()
