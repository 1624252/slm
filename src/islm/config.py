"""Central configuration: behavior thresholds (from the PRD) and LLM settings."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
EVALS_DIR = REPO_ROOT / "evals"


@dataclass(frozen=True)
class Thresholds:
    """Pass/fail thresholds for the Behavior Spec (PRD 4, 14)."""

    # Vocabulary control. Nation (2006): 98% coverage ~ 1 unknown word / 50 running words.
    max_oov_rate: float = 0.01
    min_coverage: float = 0.98
    allow_proper_nouns: bool = True

    # i+1 pacing: at most one new word per sentence.
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
