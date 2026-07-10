"""Run the eval over held-out scenarios for one "model" (any scenario -> story function)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ..config import DEFAULT_THRESHOLDS, Thresholds
from ..datagen.scenarios import Scenario
from ..llm.client import LLMClient
from ..validators import validate_story
from ..vocab.lemmatize import Lemmatizer, get_analyzer
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
    failures: list[str] = field(
        default_factory=list
    )  # tags: oov, coverage, one_new_word, recurrence
    story: str = ""
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
    progress: bool = False,
    judge_workers: int = 8,
) -> EvalSummary:
    """Two phases: (1) generate + validate each story *sequentially* (the generator is usually a
    single GPU, so parallelism there just thrashes VRAM); (2) score judge + cloze *concurrently*
    (independent network I/O — the slow part). Row order is preserved.
    """
    analyzers: dict[str, Lemmatizer] = {}
    rows: list[EvalRow] = []
    n = len(scenarios)

    # Phase 1 — sequential generation + deterministic validation (GPU-bound).
    for i, s in enumerate(scenarios, 1):
        # Reuse one analyzer per language (segmenter init can be expensive).
        lem = lemmatizer or analyzers.setdefault(s.language, get_analyzer(s.language))
        if progress:  # per-scenario line so long GPU runs aren't silent (looks like a hang)
            print(f"  [{model_name}] gen {i}/{n} {s.id} ...", flush=True)
        story = produce_story(s)
        report = validate_story(story, s.known_set(), s.target_set(), lem, thresholds)
        failures: list[str] = []
        if report.coverage.oov_rate > thresholds.max_oov_rate:
            failures.append("oov")
        if report.coverage.coverage < thresholds.min_coverage:
            failures.append("coverage")
        if not report.one_new_word.passed:
            failures.append("one_new_word")
        if not report.recurrence.passed:
            failures.append("recurrence")
        rows.append(
            EvalRow(
                scenario_id=s.id,
                hard_pass=report.hard_pass,
                oov_rate=report.coverage.oov_rate,
                coverage=report.coverage.coverage,
                one_new_word_pass=report.one_new_word.passed,
                recurrence_pass=report.recurrence.passed,
                failures=failures,
                story=story,
            )
        )

    # Phase 2 — concurrent judge + cloze scoring (network-bound). The OpenAI SDK client is
    # thread-safe, so a small thread pool turns ~2*n sequential round-trips into n/workers waves.
    if judge_client or cloze_client:
        _score_rows(
            rows, scenarios, judge_client, cloze_client, judge_workers, progress, model_name
        )
    return EvalSummary(model_name, rows)


def _score_rows(rows, scenarios, judge_client, cloze_client, workers, progress, model_name):
    """Fill each row's judge + cloze scores concurrently (independent per-row network calls)."""
    from concurrent.futures import ThreadPoolExecutor

    def score(pair):
        row, s = pair
        if judge_client:
            row.judge = judge_story(s, row.story, judge_client)
        if cloze_client:
            row.inferability = cloze_inferability(row.story, s.target_set(), cloze_client)["rate"]
        return row

    pairs = list(zip(rows, scenarios, strict=True))
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for _ in pool.map(score, pairs):
            done += 1
            if progress:
                print(f"  [{model_name}] scored {done}/{len(pairs)}", flush=True)
