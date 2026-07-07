"""Deterministic validators for the i+1 Behavior Spec (PRD 4, 14).

These are the backbone: the same checks filter the dataset, guide rewriting,
grade the eval, and guard inference. No LLM calls happen here.
"""

from .coverage import CoverageResult, coverage
from .one_new_word import OneNewWordResult, SentenceNewWords, one_new_word
from .recurrence import RecurrenceResult, recurrence
from .report import ValidationReport, validate_story

__all__ = [
    "CoverageResult",
    "OneNewWordResult",
    "RecurrenceResult",
    "SentenceNewWords",
    "ValidationReport",
    "coverage",
    "one_new_word",
    "recurrence",
    "validate_story",
]
