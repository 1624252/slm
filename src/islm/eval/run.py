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
from ..vocab.wordlists import VOCAB_DIR, Vocabulary
from .adversarial import make_adversarial_scenarios
from .generators import HFGenerator, StoryGenerator, api_generator, client_rewriter, guarded
from .harness import evaluate
from .report import error_analysis, results_markdown, single_model_markdown, summary_metrics
from .track import track_results


def _build_generator(model, path, adapter, mock, max_new_tokens, no_think):
    """Return (generator, client_or_None). The client (API/mock only) powers the guard rewriter."""
    if mock:
        client = get_client(mock=True)
        return api_generator(client=client), client
    if path:
        chat_kwargs = {"enable_thinking": False} if no_think else None
        return HFGenerator(
            path, adapter, max_new_tokens=max_new_tokens, chat_kwargs=chat_kwargs
        ), None
    client = get_client(model)
    return api_generator(client=client), client


def _rows_json(summary) -> list[dict]:
    """Per-scenario detail (incl. the generated story) for inspection / error analysis."""
    return [
        {
            "id": r.scenario_id,
            "hard_pass": r.hard_pass,
            "oov_rate": round(r.oov_rate, 3),
            "failures": r.failures,
            "story": r.story,
        }
        for r in summary.rows
    ]


def _load_or_sample(path: Path, factory: Callable[[], list[Scenario]]) -> list[Scenario]:
    if path.exists():
        return load_scenarios(path)
    scenarios = factory()
    save_scenarios(scenarios, path)
    return scenarios


def _sampler(language: str, n: int, seed: int, curated: bool):
    """Small curated-vocab scenarios (short prompts, CPU-friendly) or the full baseline."""
    if not curated:
        return lambda: sample_scenarios(n, language=language, seed=seed)
    known = Vocabulary.from_csv(VOCAB_DIR / language / "baseline.csv")
    pool = sorted(Vocabulary.from_csv(VOCAB_DIR / language / "advanced.csv").lemmas - known.lemmas)
    return lambda: sample_scenarios(n, language=language, seed=seed, known=known, target_pool=pool)


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
    p.add_argument(
        "--curated", action="store_true", help="Small curated vocab (short prompts, CPU)."
    )
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--no-think", action="store_true", help="Disable thinking mode (e.g. Qwen3).")
    p.add_argument("--out", type=Path, default=EVALS_DIR / "results")
    p.add_argument(
        "--track", action="store_true", help="Append this run to the results leaderboard."
    )
    p.add_argument("--run-label", default=None, help="Name for the tracked run (with --track).")
    p.add_argument("--dataset", default=None, help="Training dataset dir, recorded with --track.")
    p.add_argument(
        "--epochs", type=float, default=None, help="Train epochs, recorded with --track."
    )
    p.add_argument("--notes", default=None, help="Free-text note recorded with --track.")
    args = p.parse_args()

    lang = args.language
    default_name = f"heldout_small_{lang}.jsonl" if args.curated else f"heldout_{lang}.jsonl"
    held = _load_or_sample(
        args.scenarios or EVALS_DIR / "scenarios" / default_name,
        _sampler(lang, args.n, args.seed, args.curated),
    )

    base_gen, _ = _build_generator(
        args.base_model,
        args.base_path,
        args.base_adapter,
        args.mock,
        args.max_new_tokens,
        args.no_think,
    )
    has_tuned = bool(args.mock or args.tuned_model or args.tuned_path)
    tuned_gen = tuned_client = None
    if has_tuned:
        tuned_gen, tuned_client = _build_generator(
            args.tuned_model,
            args.tuned_path,
            args.tuned_adapter,
            args.mock,
            args.max_new_tokens,
            args.no_think,
        )
        if args.guard and tuned_client is not None:
            tuned_gen = guarded(tuned_gen, client_rewriter(tuned_client))
        elif args.guard:
            print("warning: --guard needs an API/mock client to rewrite; running the HF model raw.")

    judge_client = None
    if args.mock:
        judge_client = get_client(mock=True)
    elif args.judge_model:
        judge_client = get_client(args.judge_model)

    base_name = args.base_path or args.base_model or ("mock" if args.mock else "base")
    tuned_name = args.tuned_path or args.tuned_model or ("mock" if args.mock else "tuned")

    def run(name: str, scenarios: list[Scenario], gen: StoryGenerator):
        return evaluate(name, scenarios, gen, judge_client)

    base = run(base_name, held, base_gen)
    adv = adv_base = adv_tuned = tuned = None
    if args.adversarial:
        adv = _load_or_sample(
            EVALS_DIR / "scenarios" / f"adversarial_{lang}.jsonl",
            lambda: make_adversarial_scenarios(args.adv_n, language=lang, seed=args.seed),
        )
        adv_base = run(f"{base_name} (adv)", adv, base_gen)

    if has_tuned:
        tuned = run(tuned_name, held, tuned_gen)
        if adv is not None:
            adv_tuned = run(f"{tuned_name} (adv)", adv, tuned_gen)
        md = results_markdown(base, tuned, adv_base, adv_tuned)
    else:
        md = single_model_markdown(base, adv_base)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"results_{lang}.md").write_text(md, encoding="utf-8")
    payload: dict = {
        "base": summary_metrics(base),
        "error_analysis": {"base": error_analysis(base)},
        "base_rows": _rows_json(base),
    }
    if tuned is not None:
        payload["tuned"] = summary_metrics(tuned)
        payload["error_analysis"]["tuned"] = error_analysis(tuned)
        payload["tuned_rows"] = _rows_json(tuned)
    if adv_base is not None:
        payload["adversarial"] = {"base": summary_metrics(adv_base)}
        if adv_tuned is not None:
            payload["adversarial"]["tuned"] = summary_metrics(adv_tuned)
    results_path = args.out / f"results_{lang}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(md)

    if args.track:
        label = args.run_label or f"{Path(tuned_name).name if has_tuned else Path(base_name).name}"
        rec = track_results(
            results_path,
            label=label,
            language=lang,
            dataset=args.dataset,
            base_model=base_name,
            tuned_model=tuned_name if has_tuned else None,
            tuned_adapter=args.tuned_adapter,
            epochs=args.epochs,
            notes=args.notes,
        )
        print(
            f"\ntracked run '{rec.label}' -> leaderboard updated ({EVALS_DIR / 'LEADERBOARD.md'})"
        )


if __name__ == "__main__":
    main()
