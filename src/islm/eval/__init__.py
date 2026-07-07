"""Evaluation: deterministic checks + LLM judge + robustness, run base-vs-tuned over held-out sets.

The model under test is any `StoryGenerator` (API, local fine-tuned HF, or mock), so a trained
model is evaluated with a single command.
"""

from .adversarial import make_adversarial_scenarios
from .cloze import cloze_inferability
from .generators import HFGenerator, StoryGenerator, api_generator, client_rewriter, guarded
from .harness import EvalRow, EvalSummary, evaluate
from .judge import DIMENSIONS, SPEC_DIMENSIONS, judge_story
from .report import base_vs_tuned_table, error_analysis, results_markdown, summary_metrics

__all__ = [
    "DIMENSIONS",
    "SPEC_DIMENSIONS",
    "EvalRow",
    "EvalSummary",
    "HFGenerator",
    "StoryGenerator",
    "api_generator",
    "base_vs_tuned_table",
    "client_rewriter",
    "cloze_inferability",
    "error_analysis",
    "evaluate",
    "guarded",
    "judge_story",
    "make_adversarial_scenarios",
    "results_markdown",
    "summary_metrics",
]
