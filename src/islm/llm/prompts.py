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


_JUDGE_DIMENSIONS = (
    "spec_adherence",  # holds i+1 vocabulary + pacing + recurrence
    "inferability",  # new words guessable from context
    "engagement",  # compelling, worth reading
    "coherence",  # a real, connected story
    "consistency",  # stable behavior across the story
    "seductive_detail_control",  # humor carries the word; lesson not announced
)


def judge_prompt(scenario: Scenario, story: str) -> tuple[str, str]:
    system = (
        "TASK: JUDGE\n"
        "You are a strict evaluator of comprehensible-input learner stories.\n"
        "Score each dimension 0 (fails), 1 (partial), or 2 (fully). Return ONLY a JSON object "
        "with integer scores for: " + ", ".join(_JUDGE_DIMENSIONS) + ", and a short 'rationale'."
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
