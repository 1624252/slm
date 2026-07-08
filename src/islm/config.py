"""Central configuration: behavior thresholds (from the PRD) and LLM settings."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
EVALS_DIR = REPO_ROOT / "evals"

# Languages with curated word lists + committed scenarios. Evals run on all of these by default
# so no language is silently skipped; the pipeline still supports any language via fallbacks.
SHIPPED_LANGUAGES = ("en", "zh", "ja")


@dataclass(frozen=True)
class Thresholds:
    """Pass/fail thresholds for the Behavior Spec (PRD 4, 14)."""

    # Vocabulary control. Target/ideal is 100% coverage (every word in-vocabulary); a story
    # passes when OOV <= 2%. Since coverage = 1 - OOV rate, that pass gate == coverage >= 98%
    # (100% coverage and 2% OOV cannot both be hard gates). Nation (2006): 98% ~ 1 unknown / 50.
    max_oov_rate: float = 0.02
    min_coverage: float = 0.98
    allow_proper_nouns: bool = True

    # i+1 pacing: at most one new word in EVERY sentence (a story passes only if 100% of its
    # sentences comply).
    max_new_words_per_sentence: int = 1

    # Spaced repetition. SRS-Stories: >=3 per story; Waring & Takaki: ~8 across a series.
    min_recurrence: int = 3
    series_min_recurrence: int = 8

    # LLM-as-judge (0-2 per dimension).
    judge_min_spec_adherence: int = 2
    judge_min_mean: float = 1.5

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_THRESHOLDS = Thresholds()


@dataclass
class LLMConfig:
    """OpenAI-compatible client settings. Read from the environment; never hard-code keys."""

    api_key: str | None = None
    base_url: str | None = None
    teacher_model: str = "gpt-5"
    judge_model: str = "claude-sonnet-5"
    request_timeout: float = 120.0

    @classmethod
    def from_env(cls) -> LLMConfig:
        # Load .env if python-dotenv is present (optional convenience).
        try:
            from dotenv import load_dotenv

            load_dotenv(REPO_ROOT / ".env")
        except Exception:
            pass
        return cls(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            teacher_model=os.getenv("TEACHER_MODEL", "gpt-5"),
            judge_model=os.getenv("JUDGE_MODEL", "claude-sonnet-5"),
        )
