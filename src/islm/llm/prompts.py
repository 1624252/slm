"""Prompt templates for generation, rewriting, judging, and cloze inferability.

The first system line is a machine-readable ``TASK: ...`` tag. Real models simply treat it
as an instruction; the offline MockLLM uses it to route. Prompts also expose parseable
``TARGET_WORDS`` / ``KNOWN_WORDS`` lines (as in SRS-Stories, the allowed list is given).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..config import DEFAULT_THRESHOLDS, Thresholds

if TYPE_CHECKING:
    from ..datagen.scenarios import Scenario

_RULES = """You write short stories for language learners using comprehensible input
(Krashen's i+1 comprehensible input).
Rules:
- Use ONLY words from the allowed list below, plus simple proper-noun names.
- Introduce at most ONE new target word per sentence; every other sentence uses only known words.
- Make each new word's meaning inferable from the sentence around it.
- Reuse every target word at least {min_recurrence} times across the story.
- Keep it coherent and compelling; put any humor or surprise on the sentence with the new word.
- Never announce that you are teaching, and never label the target words.
- Output ONLY the story text."""


def _rules(thresholds: Thresholds) -> str:
    return _RULES.format(min_recurrence=thresholds.min_recurrence)


def _scenario_fields(scenario: Scenario) -> str:
    return (
        f"Language: {scenario.language}\n"
        f"Level: {scenario.level}\n"
        f"Theme: {scenario.theme}\n"
        f"TARGET_WORDS: {', '.join(scenario.target_words)}\n"
        f"KNOWN_WORDS: {', '.join(scenario.known)}"
    )


def generation_prompt(
    scenario: Scenario,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    min_sentences: int = 8,
    max_sentences: int = 14,
) -> tuple[str, str]:
    system = "TASK: GENERATE\n" + _rules(thresholds)
    user = f"{_scenario_fields(scenario)}\nWrite {min_sentences}-{max_sentences} sentences."
    return system, user


def rewrite_prompt(
    scenario: Scenario,
    story: str,
    failures: list[str],
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> tuple[str, str]:
    system = (
        "TASK: REWRITE\n"
        + _rules(thresholds)
        + "\nRevise the story to satisfy ALL rules. Change as little as possible."
    )
    problems = "\n".join(f"- {f}" for f in failures) or "- (none)"
    user = f"{_scenario_fields(scenario)}\n\nStory to fix:\n{story}\n\nProblems to fix:\n{problems}"
    return system, user


# Rubric dimensions scored 0/1/2. The first four mirror spec.md Appendix A; the last two are
# project-specific. SPEC_DIMENSIONS are the ones the win condition is judged on.
JUDGE_DIMENSIONS = (
    "spec_adherence",
    "robustness",
    "task_quality",
    "consistency",
    "inferability",
    "seductive_detail_control",
)
SPEC_DIMENSIONS = ("spec_adherence", "robustness", "task_quality", "consistency")

_RUBRIC = """Rubric - score each dimension 0 (fails), 1 (partial), or 2 (fully):
- spec_adherence: only allowed words, <=1 new word per sentence, repeats each target.
- robustness: holds the behavior even on hard, messy, or adversarial input.
- task_quality: a genuinely good, coherent, engaging story.
- consistency: stable behavior across the whole story.
- inferability: each new word's meaning is guessable from its context.
- seductive_detail_control: humor carries the target word; never announces the lesson."""


def judge_prompt(scenario: Scenario, story: str) -> tuple[str, str]:
    system = (
        "TASK: JUDGE\n"
        "You are a strict evaluator of comprehensible-input (i+1) learner stories.\n"
        + _RUBRIC
        + "\nReturn ONLY a JSON object with integer scores for "
        + ", ".join(JUDGE_DIMENSIONS)
        + ' plus a short "rationale".'
    )
    user = (
        f"TARGET_WORDS: {', '.join(scenario.target_words)}\n\nStory:\n{story}\n\nReturn JSON only."
    )
    return system, user


def cloze_prompt(masked_story: str) -> tuple[str, str]:
    system = (
        "TASK: CLOZE\n"
        "A single word has been replaced by ____ throughout the text. Using only the context, "
        "guess that one word. Return ONLY the word, nothing else."
    )
    return system, masked_story
