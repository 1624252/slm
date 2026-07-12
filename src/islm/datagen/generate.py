"""Generate one training example: teacher story -> validate -> rewrite loop -> optional judge."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from ..config import DEFAULT_THRESHOLDS, Thresholds
from ..llm.client import LLMClient
from ..llm.prompts import generation_prompt, rewrite_prompt
from ..validators import ValidationReport, validate_story
from ..vocab.lemmatize import Lemmatizer, get_analyzer
from ..vocab.tokenize import split_sentences
from .scenarios import Scenario

# A judge maps (scenario, story) -> {dimension: score, ..., "rationale": str}.
JudgeFn = Callable[[Scenario, str], dict]


def _judge_mean(judge: dict) -> float:
    scores = [v for v in judge.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return sum(scores) / len(scores) if scores else 0.0


@dataclass
class Example:
    scenario: Scenario
    story: str
    report: ValidationReport
    rewrite_passes: int
    judge: dict | None = None
    kept: bool = False

    def to_record(self, split: str | None = None) -> dict:
        """Serialize to the JSONL training schema (PRD 13.1)."""
        system, user = generation_prompt(self.scenario)
        return {
            "id": self.scenario.id,
            "language": self.scenario.language,
            "level": self.scenario.level,
            "theme": self.scenario.theme,
            "target_words": self.scenario.target_words,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
                {"role": "assistant", "content": self.story},
            ],
            "metadata": {
                "sentences": len(split_sentences(self.story)),
                "oov_rate": round(self.report.coverage.oov_rate, 4),
                "coverage": round(self.report.coverage.coverage, 4),
                "max_new_words_per_sentence": self.report.one_new_word.max_new_words,
                # JSON string, not a dict: the keys are the (per-record) target words, so a dict
                # column gives every record a different struct schema and breaks HF/Arrow parquet
                # conversion. A string keeps the column type stable.
                "target_recurrence": json.dumps(self.report.recurrence.counts, ensure_ascii=False),
                "rewrite_passes": self.rewrite_passes,
                "judge_scores": self.judge,
                "hard_pass": self.report.hard_pass,
                "kept": self.kept,
                "split": split,
            },
        }


def generate_story(
    scenario: Scenario,
    client: LLMClient,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    temperature: float = 0.7,
) -> str:
    system, user = generation_prompt(scenario, thresholds)
    return client.complete(system, user, temperature=temperature, max_tokens=800)


def rewrite_story(
    scenario: Scenario,
    story: str,
    failures: list[str],
    client: LLMClient,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    temperature: float = 0.4,
) -> str:
    system, user = rewrite_prompt(scenario, story, failures, thresholds)
    return client.complete(system, user, temperature=temperature, max_tokens=800)


def make_example(
    scenario: Scenario,
    client: LLMClient,
    lemmatizer: Lemmatizer | None = None,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    max_rewrites: int = 5,
    judge_fn: JudgeFn | None = None,
    temperature: float = 0.7,
) -> Example:
    """Generate, then run the validator-guided rewrite loop, then optionally judge."""
    lemmatizer = lemmatizer or get_analyzer(scenario.language)
    known, target = scenario.known_set(), scenario.target_set()

    story = generate_story(scenario, client, thresholds, temperature)
    report = validate_story(story, known, target, lemmatizer, thresholds)

    passes = 0
    while not report.hard_pass and passes < max_rewrites:
        story = rewrite_story(scenario, story, report.failures(), client, thresholds)
        report = validate_story(story, known, target, lemmatizer, thresholds)
        passes += 1

    judge = judge_fn(scenario, story) if judge_fn else None
    judge_ok = True
    if judge is not None:
        judge_ok = (
            judge.get("spec_adherence", 0) >= thresholds.judge_min_spec_adherence
            and _judge_mean(judge) >= thresholds.judge_min_mean
        )

    kept = report.hard_pass and judge_ok
    return Example(scenario, story, report, passes, judge, kept)
