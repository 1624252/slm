"""Scenarios: (level, known vocabulary, target words, theme) — the input the model writes from."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from ..vocab.wordlists import Vocabulary

# Candidate words to teach. The sampler drops any that are already in the known list.
DEFAULT_TARGET_POOL = [
    "clue",
    "whisker",
    "shadow",
    "treasure",
    "secret",
    "umbrella",
    "lantern",
    "puzzle",
    "ribbon",
    "feather",
    "pumpkin",
    "whistle",
    "balloon",
    "kitten",
    "mustache",
    "castle",
    "dragon",
    "wizard",
    "ghost",
    "robot",
    "pocket",
    "button",
    "candle",
    "basket",
]

# Narrow-reading themes (Krashen 2004): recurring worlds that recycle vocabulary.
THEMES = [
    "a cat detective",
    "a lost key",
    "a funny dog",
    "a quiet garden",
    "a rainy day",
    "a new friend",
    "a night in the old house",
    "a bird and a tree",
    "a mouse and the cheese",
    "a walk in the park",
]


@dataclass
class Scenario:
    id: str
    language: str
    level: str
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
    level: str = "A2",
    seed: int = 0,
    language: str = "en",
    vocab: Vocabulary | None = None,
    target_pool: list[str] | None = None,
    max_targets: int = 2,
) -> list[Scenario]:
    """Deterministically sample `n` scenarios for a level (seeded)."""
    rng = random.Random(seed)
    vocab = vocab or Vocabulary.from_csv(level_at_most=level)
    known = sorted(vocab.lemmas)
    pool = [w for w in (target_pool or DEFAULT_TARGET_POOL) if w.lower() not in vocab.lemmas]
    if not pool:
        raise ValueError("Target pool is empty after removing known words.")

    scenarios: list[Scenario] = []
    for i in range(n):
        k = rng.randint(1, max_targets)
        targets = rng.sample(pool, min(k, len(pool)))
        theme = rng.choice(THEMES)
        scenarios.append(
            Scenario(f"{language}-{level}-{i:04d}", language, level, theme, targets, known)
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
