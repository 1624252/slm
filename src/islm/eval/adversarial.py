"""Adversarial scenarios that stress the i+1 behavior (PRD 14.5 / spec Appendix A "Robustness").

Each scenario combines three pressures that tempt a model off-spec: a **small** known vocabulary
(the committed curated tier), **multiple, harder** target words, and a **jargon-tempting theme**.
A robust model still stays in-vocabulary and paces new words; a brittle one leaks OOV words.
"""

from __future__ import annotations

import random

from ..datagen.scenarios import Scenario
from ..vocab.wordlists import VOCAB_DIR, Vocabulary, load_advanced, load_baseline

_THEMES = [
    "an advanced academic lecture about the economy",
    "a technical manual full of jargon",
    "a philosophical essay for adults",
    "a news report about international politics",
    "a science article about astrophysics",
]


def make_adversarial_scenarios(
    n: int, language: str = "en", seed: int = 7, max_targets: int = 3
) -> list[Scenario]:
    rng = random.Random(seed)
    # Small known set = the committed curated tier (a realistic beginner vocabulary).
    curated = VOCAB_DIR / language / "baseline.csv"
    known_vocab = Vocabulary.from_csv(curated) if curated.exists() else load_baseline(language)
    known = sorted(known_vocab.lemmas)
    pool = sorted(load_advanced(language).lemmas - known_vocab.lemmas)
    if not pool:
        raise ValueError(f"No adversarial targets available for '{language}'.")

    scenarios: list[Scenario] = []
    for i in range(n):
        k = rng.randint(2, max(2, max_targets))
        targets = rng.sample(pool, min(k, len(pool)))
        theme = rng.choice(_THEMES)
        scenarios.append(
            Scenario(f"{language}-adv-{i:04d}", language, "adversarial", theme, targets, known)
        )
    return scenarios
