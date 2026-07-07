"""i+1 Story SLM: comprehensible-input story generation, validators, and evaluation.

The validators are the backbone: the same deterministic checks filter the dataset,
guide the rewrite loop, grade the eval, and guard inference.
"""

from .config import DEFAULT_THRESHOLDS, LLMConfig, Thresholds
from .validators.report import ValidationReport, validate_story

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_THRESHOLDS",
    "LLMConfig",
    "Thresholds",
    "ValidationReport",
    "validate_story",
    "__version__",
]
