"""Evaluation: deterministic checks + LLM judge, run base-vs-tuned over held-out scenarios."""

from .cloze import cloze_inferability
from .harness import EvalRow, EvalSummary, evaluate
from .judge import DIMENSIONS, judge_story
from .report import base_vs_tuned_table, summary_metrics

__all__ = [
    "DIMENSIONS",
    "EvalRow",
    "EvalSummary",
    "base_vs_tuned_table",
    "cloze_inferability",
    "evaluate",
    "judge_story",
    "summary_metrics",
]
