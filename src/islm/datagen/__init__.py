"""Data generation: scenario sampling, teacher generation + rewriting, second-pass curation.

The ``curate``, ``pipeline``, and ``seed`` modules are runnable with ``python -m`` and are
imported directly (e.g. ``from islm.datagen.curate import curate``) rather than eagerly here.
"""

from .generate import Example, generate_story, make_example, rewrite_story
from .scenarios import Scenario, load_scenarios, sample_scenarios, save_scenarios

__all__ = [
    "Example",
    "Scenario",
    "generate_story",
    "load_scenarios",
    "make_example",
    "rewrite_story",
    "sample_scenarios",
    "save_scenarios",
]
