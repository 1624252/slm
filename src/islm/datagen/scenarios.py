"""Scenarios: (language, known vocabulary, target words, theme) — the model's input."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from ..vocab.languages import get_language
from ..vocab.wordlists import Vocabulary, load_advanced, load_baseline


@dataclass
class Scenario:
    id: str
    language: str
    level: str  # human label for the known tier(s), e.g. "A1-A2", "HSK1-HSK3", "N5-N4"
    theme: str
    target_words: list[str]
    known: list[str]

    def known_set(self) -> set[str]:
        return {w.lower() for w in self.known}

    def target_set(self) -> set[str]:
        return {w.lower() for w in self.target_words}

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Scenario:
        return cls(**data)


def sample_scenarios(
    n: int,
    language: str = "en",
    seed: int = 0,
    max_targets: int = 2,
    known: Vocabulary | None = None,
    target_pool: list[str] | None = None,
) -> list[Scenario]:
    """Deterministically sample `n` scenarios for a language (seeded).

    Known vocabulary = the language's baseline tier; target pool = its advanced tier
    (minus anything already known). Falls back to frequency bands for unlisted languages.
    """
    lang = get_language(language)
    known_vocab = known or load_baseline(language)
    known_list = sorted(known_vocab.lemmas)
    pool = target_pool or sorted(load_advanced(language).lemmas - known_vocab.lemmas)
    if not pool:
        raise ValueError(f"No target words available for language '{language}'.")

    level = "-".join(lang.baseline_tiers) or lang.level_scheme
    rng = random.Random(seed)
    scenarios: list[Scenario] = []
    for i in range(n):
        k = rng.randint(1, max_targets)
        targets = rng.sample(pool, min(k, len(pool)))
        theme = rng.choice(lang.themes)
        scenarios.append(
            Scenario(f"{language}-{i:04d}", language, level, theme, targets, known_list)
        )
    return scenarios


def save_scenarios(scenarios: list[Scenario], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in scenarios:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")


def load_scenarios(path: Path) -> list[Scenario]:
    with open(path, encoding="utf-8") as f:
        return [Scenario.from_dict(json.loads(line)) for line in f if line.strip()]
