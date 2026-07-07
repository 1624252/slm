"""Run the eval over held-out scenarios for one "model" (any scenario -> story function)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ..config import DEFAULT_THRESHOLDS, Thresholds
from ..datagen.scenarios import Scenario
from ..llm.client import LLMClient
from ..validators import validate_story
from ..vocab.lemmatize import Lemmatizer, get_lemmatizer
from .cloze import cloze_inferability
from .judge import DIMENSIONS, judge_story

# A model under test is anything that turns a scenario into a story (base, tuned, or mock).
StoryProducer = Callable[[Scenario], str]


@dataclass
class EvalRow:
    scenario_id: str
    hard_pass: bool
    oov_rate: float
    coverage: float
    one_new_word_pass: bool
    recurrence_pass: bool
    judge: dict | None = None
    inferability: float | None = None


@dataclass
class EvalSummary:
    model: str
    rows: list[EvalRow] = field(default_factory=list)

    def aggregate(self) -> dict:
        rows = self.rows
        n = len(rows)
        if n == 0:
            return {"model": self.model, "n": 0}

        def rate(values) -> float:
            return round(sum(values) / n, 4)

        metrics = {
            "model": self.model,
            "n": n,
            "hard_pass_rate": rate(r.hard_pass for r in rows),
            "mean_oov_rate": rate(r.oov_rate for r in rows),
            "mean_coverage": rate(r.coverage for r in rows),
            "one_new_word_pass_rate": rate(r.one_new_word_pass for r in rows),
            "recurrence_pass_rate": rate(r.recurrence_pass for r in rows),
        }

        judged = [r.judge for r in rows if r.judge]
        if judged:
            for dim in DIMENSIONS:
                metrics[f"judge_{dim}"] = round(sum(j[dim] for j in judged) / len(judged), 3)
            metrics["judge_overall"] = round(
                sum(metrics[f"judge_{dim}"] for dim in DIMENSIONS) / len(DIMENSIONS), 3
            )

        infer = [r.inferability for r in rows if r.inferability is not None]
        if infer:
            metrics["mean_inferability"] = round(sum(infer) / len(infer), 3)
        return metrics


def evaluate(
    model_name: str,
    scenarios: list[Scenario],
    produce_story: StoryProducer,
    judge_client: LLMClient | None = None,
    cloze_client: LLMClient | None = None,
    lemmatizer: Lemmatizer | None = None,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> EvalSummary:
    lemmatizer = lemmatizer or get_lemmatizer()
    rows: list[EvalRow] = []
    for s in scenarios:
        story = produce_story(s)
        report = validate_story(story, s.known_set(), s.target_set(), lemmatizer, thresholds)
        rows.append(
            EvalRow(
                scenario_id=s.id,
                hard_pass=report.hard_pass,
                oov_rate=report.coverage.oov_rate,
                coverage=report.coverage.coverage,
                one_new_word_pass=report.one_new_word.passed,
                recurrence_pass=report.recurrence.passed,
                judge=judge_story(s, story, judge_client) if judge_client else None,
                inferability=(
                    cloze_inferability(story, s.target_set(), cloze_client)["rate"]
                    if cloze_client
                    else None
                ),
            )
        )
    return EvalSummary(model_name, rows)
