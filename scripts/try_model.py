"""Quick manual test of a trained i+1 model: pick a mode, see the sampled vocab, read the story.

Four modes (the target words are what the model must teach; known = the allowed palette):
  en       — English:  known = baseline (A1-A2),          targets = 2 random advanced (B1-C1)
  en-exam  — English:  known = baseline + advanced,       targets = 2 random exam (GRE/SAT/ACT)
  zh       — Chinese:  known = baseline (HSK1-3),          targets = 2 random advanced (HSK4-6)
  jp       — Japanese: known = baseline (N5-N4),           targets = 2 random advanced

The random target/theme selection is printed BEFORE the model runs, so you see the inputs first.

    # base model only (no fine-tune):
    python scripts/try_model.py --mode en --base-path Qwen/Qwen3-4B-Instruct-2507 --no-think
    # a trained adapter on top of the base:
    python scripts/try_model.py --mode en-exam --base-path Qwen/Qwen3-4B-Instruct-2507 \
        --adapter /content/drive/MyDrive/islm_v2_multi/qwen3_4b_v2_multi --no-think
    # zh / jp:
    python scripts/try_model.py --mode zh  --base-path <path> --adapter <path> --no-think
    python scripts/try_model.py --mode jp  --base-path <path> --adapter <path> --no-think
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from islm.datagen.scenarios import Scenario
from islm.vocab.languages import get_language
from islm.vocab.wordlists import VOCAB_DIR, Vocabulary

MODES = {
    # mode -> (language, exam?)
    "en": ("en", False),
    "en-exam": ("en", True),
    "zh": ("zh", False),
    "jp": ("ja", False),
}


def _load(language: str, name: str) -> Vocabulary:
    path = VOCAB_DIR / language / f"{name}.csv"
    if not path.exists():
        raise SystemExit(f"missing vocab: {path}")
    return Vocabulary.from_csv(path)


def build_scenario(mode: str, seed: int) -> Scenario:
    """Sample the known palette + 2 random targets for the mode; return a ready-to-run Scenario."""
    language, exam = MODES[mode]
    rng = random.Random(seed)
    lang = get_language(language)

    baseline = _load(language, "baseline")
    advanced = _load(language, "advanced")

    if exam:
        # known = baseline + advanced; targets = 2 random exam words (English only)
        known_vocab = baseline | advanced
        exam_vocab = _load(language, "exam")
        pool = sorted(exam_vocab.lemmas - known_vocab.lemmas)
        level = f"{'+'.join(lang.baseline_tiers)} + {'+'.join(lang.advanced_tiers)}"
        target_tier = "exam (GRE/SAT/ACT)"
    else:
        # known = baseline; targets = 2 random advanced words
        known_vocab = baseline
        pool = sorted(advanced.lemmas - baseline.lemmas)
        level = "+".join(lang.baseline_tiers) or "baseline"
        target_tier = "+".join(lang.advanced_tiers) or "advanced"

    if len(pool) < 2:
        raise SystemExit(f"target pool for {mode} has <2 words ({len(pool)})")
    targets = rng.sample(pool, 2)
    theme = rng.choice(list(lang.themes) or ["a small adventure"])

    scenario = Scenario(
        id=f"try-{mode}-{seed}",
        language=language,
        level=level,
        theme=theme,
        target_words=targets,
        known=sorted(known_vocab.lemmas),
    )
    # Show the selection before running the model.
    print("=" * 70)
    print(f"MODE        : {mode}  (language={language})")
    print(f"KNOWN tier  : {level}  ({len(scenario.known)} words allowed)")
    print(f"TARGET tier : {target_tier}  (pool size {len(pool)})")
    print(f"TARGETS     : {targets}   <-- the 2 words the model must teach")
    print(f"THEME       : {theme}")
    print("=" * 70)
    return scenario


def main() -> None:
    p = argparse.ArgumentParser(description="Manually test a trained i+1 story model.")
    p.add_argument("--mode", required=True, choices=list(MODES), help="en | en-exam | zh | jp")
    p.add_argument("--base-path", required=True, help="HF model path/id for the base.")
    p.add_argument("--adapter", default=None, help="LoRA adapter path (omit to test the base).")
    p.add_argument("--seed", type=int, default=None, help="Fix the selection (default: random).")
    p.add_argument("--max-new-tokens", type=int, default=320)
    p.add_argument("--no-think", action="store_true", help="Disable thinking mode (e.g. Qwen3).")
    p.add_argument("--temperature", type=float, default=0.0)
    args = p.parse_args()

    # A fresh random seed each run unless pinned, so repeated calls vary.
    seed = args.seed if args.seed is not None else random.randint(0, 10_000)
    scenario = build_scenario(args.mode, seed)

    print("\nloading model (this can take a minute the first time)...", flush=True)
    import torch

    from islm.eval.generators import HFGenerator

    on_gpu = torch.cuda.is_available()
    gen = HFGenerator(
        args.base_path,
        args.adapter,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device_map="auto" if on_gpu else None,
        load_in_4bit=on_gpu,
        chat_kwargs={"enable_thinking": False} if args.no_think else None,
    )
    label = f"base + adapter ({args.adapter})" if args.adapter else "base (no adapter)"
    print(f"model       : {args.base_path}  [{label}]  device={'gpu' if on_gpu else 'cpu'}\n")

    story = gen(scenario)
    print("-" * 70)
    print("GENERATED STORY:\n")
    print(story)
    print("-" * 70)
    print(f"(seed={seed} — pass --seed {seed} to reproduce this exact selection)")


if __name__ == "__main__":
    main()
