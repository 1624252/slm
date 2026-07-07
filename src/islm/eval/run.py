"""Run the base-vs-tuned evaluation over held-out scenarios.

Offline smoke (mock as both models):

    python -m islm.eval.run --mock

Real comparison (prompted base vs your fine-tuned model, OpenAI-compatible endpoints):

    python -m islm.eval.run --base-model qwen3-4b-instruct \
        --tuned-model my-tuned --judge-model gpt-5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import EVALS_DIR
from ..datagen.generate import generate_story
from ..datagen.scenarios import load_scenarios, sample_scenarios, save_scenarios
from ..llm.client import LLMClient, get_client
from .harness import evaluate
from .report import base_vs_tuned_table, summary_metrics


def _producer(client: LLMClient):
    # Temperature 0 for reproducible eval runs.
    return lambda scenario: generate_story(scenario, client, temperature=0.0)


def main() -> None:
    p = argparse.ArgumentParser(description="Base-vs-tuned eval for the i+1 story model.")
    p.add_argument("--scenarios", type=Path, default=EVALS_DIR / "scenarios" / "heldout.jsonl")
    p.add_argument("--n", type=int, default=12, help="Sampled if the scenarios file is missing.")
    p.add_argument("--level", default="A2")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--base-model", default=None)
    p.add_argument("--tuned-model", default=None)
    p.add_argument("--judge-model", default=None, help="Omit to skip LLM judging.")
    p.add_argument("--mock", action="store_true", help="Use the offline MockLLM for all roles.")
    p.add_argument("--out", type=Path, default=EVALS_DIR / "results")
    args = p.parse_args()

    if args.scenarios.exists():
        scenarios = load_scenarios(args.scenarios)
    else:
        scenarios = sample_scenarios(args.n, level=args.level, seed=args.seed)
        save_scenarios(scenarios, args.scenarios)

    if args.mock:
        base_client = tuned_client = judge_client = get_client(mock=True)
    else:
        base_client = get_client(args.base_model)
        tuned_client = get_client(args.tuned_model)
        judge_client = get_client(args.judge_model) if args.judge_model else None

    base = evaluate("base", scenarios, _producer(base_client), judge_client)
    tuned = evaluate("tuned", scenarios, _producer(tuned_client), judge_client)

    table = base_vs_tuned_table(base, tuned)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.md").write_text(table + "\n", encoding="utf-8")
    with open(args.out / "results.json", "w", encoding="utf-8") as f:
        json.dump({"base": summary_metrics(base), "tuned": summary_metrics(tuned)}, f, indent=2)
    print(table)


if __name__ == "__main__":
    main()
