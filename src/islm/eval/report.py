"""Turn eval summaries into the base-vs-tuned results report (spec Appendix A required outputs)."""

from __future__ import annotations

from collections import Counter

from ..config import DEFAULT_THRESHOLDS as _TH
from .harness import EvalSummary
from .judge import DIMENSIONS

# Target (ideal) value per criterion, from the Behavior-Spec thresholds (config.Thresholds).
# Shown in table headers so each number is read against its goal.
_OOV_TARGET = f"target <={_TH.max_oov_rate:.2f}"  # lower is better
_JUDGE_TARGET = f"target >={_TH.judge_min_mean:.1f}"  # mean rubric score

# Deterministic behavioral checks (PRD 14.2). (metric key, label-with-target, higher-is-better)
_HARD_CHECKS = [
    ("hard_pass_rate", "Hard-check pass rate (target 1.000)", True),
    ("mean_oov_rate", f"OOV rate ({_OOV_TARGET})", False),
    ("one_new_word_pass_rate", "<=1 new word/sentence (target 1.000)", True),
    ("recurrence_pass_rate", "Recurrence satisfied (target 1.000)", True),
    ("mean_inferability", "Inferability (cloze; target 1.000)", True),
]

_LEGEND = (
    "\n---\n_Legend: rates are fractions in [0,1]; judge scores in [0,2] (0 = fails, 1 = partial, "
    "2 = fully). Hard-check pass rate, <=1-new-word, recurrence, coverage: **higher is better** "
    "(1.0 = every story passes). OOV rate: **lower is better** (0 = ideal; gate <= 0.02). "
    "OOV = out-of-vocabulary (a word not in the learner's known K or target T set); "
    "coverage = 1 - OOV rate._"
)


def summary_metrics(summary: EvalSummary) -> dict:
    return summary.aggregate()


def error_analysis(summary: EvalSummary) -> dict:
    """Tally which deterministic checks fail across a model's outputs (spec error-analysis)."""
    counts: Counter = Counter()
    failing = 0
    for row in summary.rows:
        if row.failures:
            failing += 1
            counts.update(row.failures)
    return {"n": len(summary.rows), "failing": failing, "reasons": dict(counts.most_common())}


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return "-" if value is None else str(value)


def _rows(base: dict, tuned: dict, specs) -> list[str]:
    out = []
    for key, label, higher in specs:
        if key not in base and key not in tuned:
            continue
        bv, tv = base.get(key), tuned.get(key)
        line = f"| {label} | {_fmt(bv)} | {_fmt(tv)} |"
        if isinstance(bv, (int, float)) and isinstance(tv, (int, float)):
            delta = tv - bv
            better = "tuned" if (delta > 0 if higher else delta < 0) else "base"
            line += f" {delta:+.3f} | {better} |"
        else:
            line += " - | - |"
        out.append(line)
    return out


def base_vs_tuned_table(base: EvalSummary, tuned: EvalSummary) -> str:
    """Compact base-vs-tuned table (hard checks + judge dimensions)."""
    b, t = base.aggregate(), tuned.aggregate()
    specs = _HARD_CHECKS + [
        (f"judge_{d}", f"Judge: {d} ({_JUDGE_TARGET})", True) for d in DIMENSIONS
    ]
    header = [
        f"| Metric | Base ({base.model}) | Tuned ({tuned.model}) | Delta | Better |",
        "| --- | --- | --- | --- | --- |",
    ]
    return "\n".join(header + _rows(b, t, specs))


def _win_condition(b: dict, t: dict, adv_b: dict | None, adv_t: dict | None) -> tuple[bool, str]:
    """Spec win: tuned beats base on Spec adherence AND Robustness."""
    spec_key = "judge_spec_adherence" if "judge_spec_adherence" in t else "hard_pass_rate"
    spec_win = t.get(spec_key, 0) > b.get(spec_key, 0)
    if adv_b is not None and adv_t is not None:
        rob_b, rob_t = adv_b.get("hard_pass_rate", 0), adv_t.get("hard_pass_rate", 0)
    else:
        rob_key = "judge_robustness" if "judge_robustness" in t else "hard_pass_rate"
        rob_b, rob_t = b.get(rob_key, 0), t.get(rob_key, 0)
    rob_win = rob_t > rob_b
    return (spec_win and rob_win), (
        f"spec-adherence {'up' if spec_win else 'not up'} ({spec_key}), "
        f"robustness {'up' if rob_win else 'not up'}"
    )


def single_model_markdown(summary: EvalSummary, adv: EvalSummary | None = None) -> str:
    """Report one model's metrics + error analysis (e.g. a Day-1 base-model baseline)."""
    agg = summary.aggregate()
    lines = [
        f"# Eval: {summary.model}",
        f"\nHeld-out scenarios: **{agg.get('n', 0)}**.",
        "\n## Behavioral checks (deterministic — the failures the spec forbids)",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    for key, label, _ in _HARD_CHECKS:
        if key in agg:
            lines.append(f"| {label} | {_fmt(agg[key])} |")
    judged = [f"judge_{d}" for d in DIMENSIONS if f"judge_{d}" in agg]
    if judged:
        lines += ["\n## LLM-as-judge rubric (0-2)", "| Dimension | Mean |", "| --- | --- |"]
        lines += [f"| {k.removeprefix('judge_')} | {_fmt(agg[k])} |" for k in judged]
    if adv is not None:
        aagg = adv.aggregate()
        lines += [
            f"\n## Robustness (adversarial, n={aagg.get('n', 0)})",
            "| Metric | Value |",
            "| --- | --- |",
            f"| Adversarial hard-check pass | {_fmt(aagg.get('hard_pass_rate'))} |",
        ]
    ea = error_analysis(summary)
    lines.append("\n## Error analysis")
    if ea["reasons"]:
        lines.append(f"{ea['failing']}/{ea['n']} outputs failed a check. Most common:")
        lines += [f"- `{reason}`: {count}" for reason, count in ea["reasons"].items()]
    else:
        lines.append(f"All {ea['n']} outputs passed the deterministic checks.")
    return "\n".join(lines) + _LEGEND + "\n"


def results_markdown(
    base: EvalSummary,
    tuned: EvalSummary,
    adv_base: EvalSummary | None = None,
    adv_tuned: EvalSummary | None = None,
) -> str:
    """Full results doc: hard checks, judge rubric, robustness, win verdict, and error analysis."""
    b, t = base.aggregate(), tuned.aggregate()
    hdr = ["| Metric | Base | Tuned | Delta | Better |", "| --- | --- | --- | --- | --- |"]

    lines = [
        f"# Eval results: base ({base.model}) vs tuned ({tuned.model})",
        f"\nHeld-out scenarios: **{b.get('n', 0)}**.",
        "\n## Behavioral checks (deterministic — the failures the spec forbids)",
        *hdr,
        *_rows(b, t, _HARD_CHECKS),
        "\n## LLM-as-judge rubric (0-2; first four are spec Appendix A)",
        *hdr,
        *_rows(b, t, [(f"judge_{d}", f"{d} ({_JUDGE_TARGET})", True) for d in DIMENSIONS]),
    ]

    if adv_base is not None and adv_tuned is not None:
        ab, at = adv_base.aggregate(), adv_tuned.aggregate()
        lines += [
            f"\n## Robustness (adversarial set: tiny vocab + jargon themes, n={ab.get('n', 0)})",
            *hdr,
            *_rows(ab, at, [("hard_pass_rate", "Adversarial hard-check pass", True)]),
        ]
    else:
        ab = at = None

    won, why = _win_condition(b, t, ab, at)
    verdict = "PASS" if won else "FAIL"
    lines += [
        "\n## Win condition (spec)",
        f"Beats base on Spec adherence AND Robustness: **{verdict}** ({why}).",
        "\n## Error analysis (tuned, held-out)",
    ]
    ea = error_analysis(tuned)
    if ea["reasons"]:
        lines.append(f"{ea['failing']}/{ea['n']} outputs failed a check. Most common:")
        lines += [f"- `{reason}`: {count}" for reason, count in ea["reasons"].items()]
    else:
        lines.append(f"All {ea['n']} tuned outputs passed the deterministic checks.")
    lines.append(
        "\n_Fill in: are the remaining failures a data problem (e.g. under-represented targets, "
        "themes that tempt off-vocab words)? What data change would fix them?_"
    )
    return "\n".join(lines) + _LEGEND + "\n"
