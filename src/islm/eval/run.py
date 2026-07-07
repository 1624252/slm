"""Run the base-vs-tuned evaluation over held-out (and optional adversarial) scenarios.

Offline smoke (mock for every role):

    python -m islm.eval.run --mock --language zh --adversarial

API models (OpenAI-compatible base_url):

    python -m islm.eval.run --base-model qwen3-4b-instruct --tuned-model my-tuned \
        --judge-model gpt-5 --adversarial

Local fine-tuned checkpoint — the "once we have a trained model" path:

    python -m islm.eval.run --base-path Qwen/Qwen3-4B-Instruct \
        --tuned-path Qwen/Qwen3-4B-Instruct --tuned-adapter outputs/lora \
        --judge-model gpt-5 --adversarial

Results are written to evals/results/results_<lang>.{md,json}.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

from ..config import EVALS_DIR
from ..datagen.scenarios import Scenario, load_scenarios, sample_scenarios, save_scenarios
from ..llm.client import get_client
from .adversarial import make_adversarial_scenarios
from .generators import HFGenerator, StoryGenerator, api_generator, client_rewriter, guarded
from .harness import evaluate
from .report import error_analysis, results_markdown, summary_metrics


def _build_generator(model, path, adapter, mock):
    """Return (generator, client_or_None). The client (API/mock only) powers the guard rewriter."""
    if mock:
        client = get_client(mock=True)
        return api_generator(client=client), client
    if path:
        return HFGenerator(path, adapter), None  # local HF; no client for guard rewriting
    client = get_client(model)
    return api_generator(client=client), client


def _load_or_sample(path: Path, factory: Callable[[], list[Scenario]]) -> list[Scenario]:
    if path.exists():
        return load_scenarios(path)
    scenarios = factory()
    save_scenarios(scenarios, path)
    return scenarios


def main() -> None:
    p = argparse.ArgumentParser(description="Base-vs-tuned eval for the i+1 story model.")
    p.add_argument("--language", default="en")
    p.add_argument(
        "--scenarios", type=Path, default=None, help="Held-out JSONL (else per-lang default)."
    )
    p.add_argument("--n", type=int, default=24, help="Sampled if the held-out file is missing.")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--base-model", default=None, help="API model name for the base.")
    p.add_argument("--tuned-model", default=None, help="API model name for the tuned model.")
    p.add_argument("--base-path", default=None, help="Local HF path for the base.")
    p.add_argument("--tuned-path", default=None, help="Local HF path for the tuned model.")
    p.add_argument("--base-adapter", default=None, help="LoRA adapter path for the base.")
    p.add_argument("--tuned-adapter", default=None, help="LoRA adapter path for the tuned model.")
    p.add_argument("--judge-model", default=None, help="API judge model; omit to skip judging.")
    p.add_argument("--mock", action="store_true", help="Use the offline MockLLM for every role.")
    p.add_argument("--adversarial", action="store_true", help="Also run the adversarial set.")
    p.add_argument("--adv-n", type=int, default=12)
    p.add_argument(
        "--guard", action="store_true", help="Wrap the tuned model with the inference guard."
    )
    p.add_argument("--out", type=Path, default=EVALS_DIR / "results")
    args = p.parse_args()

    lang = args.language
    held = _load_or_sample(
        args.scenarios or EVALS_DIR / "scenarios" / f"heldout_{lang}.jsonl",
        lambda: sample_scenarios(args.n, language=lang, seed=args.seed),
    )

    base_gen, _ = _build_generator(args.base_model, args.base_path, args.base_adapter, args.mock)
    tuned_gen, tuned_client = _build_generator(
        args.tuned_model, args.tuned_path, args.tuned_adapter, args.mock
    )
    if args.guard:
        if tuned_client is None:
            print("warning: --guard needs an API/mock client to rewrite; running the HF model raw.")
        else:
            tuned_gen = guarded(tuned_gen, client_rewriter(tuned_client))

    judge_client = None
    if args.mock:
        judge_client = get_client(mock=True)
    elif args.judge_model:
        judge_client = get_client(args.judge_model)

    def run(name: str, scenarios: list[Scenario], gen: StoryGenerator):
        return evaluate(name, scenarios, gen, judge_client)

    base, tuned = run("base", held, base_gen), run("tuned", held, tuned_gen)

    adv_base = adv_tuned = None
    if args.adversarial:
        adv = _load_or_sample(
            EVALS_DIR / "scenarios" / f"adversarial_{lang}.jsonl",
            lambda: make_adversarial_scenarios(args.adv_n, language=lang, seed=args.seed),
        )
        adv_base, adv_tuned = run("base-adv", adv, base_gen), run("tuned-adv", adv, tuned_gen)

    md = results_markdown(base, tuned, adv_base, adv_tuned)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"results_{lang}.md").write_text(md, encoding="utf-8")
    payload = {
        "base": summary_metrics(base),
        "tuned": summary_metrics(tuned),
        "error_analysis": {"base": error_analysis(base), "tuned": error_analysis(tuned)},
    }
    if adv_base is not None:
        payload["adversarial"] = {
            "base": summary_metrics(adv_base),
            "tuned": summary_metrics(adv_tuned),
        }
    with open(args.out / f"results_{lang}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(md)


if __name__ == "__main__":
    main()
