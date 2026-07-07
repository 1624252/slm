"""Turn eval summaries into the base-vs-tuned results table (PRD 14.4)."""

from __future__ import annotations

from .harness import EvalSummary

# (metric key, display label, "higher is better")
_HEADLINE = [
    ("hard_pass_rate", "Hard-check pass rate", True),
    ("mean_oov_rate", "OOV rate", False),
    ("one_new_word_pass_rate", "<=1 new word/sentence", True),
    ("recurrence_pass_rate", "Recurrence satisfied", True),
    ("judge_spec_adherence", "Judge: spec adherence", True),
    ("judge_overall", "Judge: overall", True),
    ("mean_inferability", "Inferability (cloze)", True),
]


def summary_metrics(summary: EvalSummary) -> dict:
    return summary.aggregate()


def _fmt(value) -> str:
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def base_vs_tuned_table(base: EvalSummary, tuned: EvalSummary) -> str:
    """Markdown table comparing base vs tuned with deltas (win = tuned beats base)."""
    b, t = base.aggregate(), tuned.aggregate()
    lines = [
        f"| Metric | Base ({base.model}) | Tuned ({tuned.model}) | Delta | Better |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key, label, higher_better in _HEADLINE:
        if key not in b and key not in t:
            continue
        bv, tv = b.get(key), t.get(key)
        row = f"| {label} | {_fmt(bv)} | {_fmt(tv)} |"
        if isinstance(bv, (int, float)) and isinstance(tv, (int, float)):
            delta = tv - bv
            improved = delta > 0 if higher_better else delta < 0
            arrow = "^" if higher_better else "v"
            row += f" {delta:+.3f} | {'tuned' if improved else 'base'} ({arrow}) |"
        else:
            row += " - | - |"
        lines.append(row)
    return "\n".join(lines)
