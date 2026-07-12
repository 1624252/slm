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
    # base vs tuned on the SAME random scenario, with the improvement on each metric:
    python scripts/try_model.py --mode en --base-path <path> --adapter <path> --no-think --compare
"""

from __future__ import annotations

import argparse
import random
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from islm.datagen.scenarios import Scenario
from islm.vocab.languages import get_language
from islm.vocab.wordlists import VOCAB_DIR, Vocabulary

WRAP_WIDTH = 88


def _char_width(ch: str) -> int:
    """Terminal columns a char occupies: CJK/wide glyphs take 2, everything else 1."""
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def _str_width(s: str) -> int:
    """Total terminal columns for a string (CJK glyphs count as 2)."""
    return sum(_char_width(c) for c in s)


def _break_token(tok: str, width: int) -> list[str]:
    """Break a single space-less token on character boundaries (for CJK runs longer than width)."""
    parts, cur, cur_w = [], "", 0
    for ch in tok:
        w = _char_width(ch)
        if cur_w + w > width and cur:
            parts.append(cur)
            cur, cur_w = "", 0
        cur += ch
        cur_w += w
    if cur:
        parts.append(cur)
    return parts


def _wrap(text: str, width: int = WRAP_WIDTH) -> str:
    """Wrap a story to a readable width for terminal printing, breaking on WORD boundaries. A word
    is only split mid-token when it alone exceeds the width (e.g. a space-less CJK run), so Chinese
    and Japanese wrap too. CJK glyphs count as 2 columns. Preserves the story's own line breaks."""
    lines = []
    for para in text.splitlines() or [text]:
        if not para.strip():
            lines.append("")
            continue
        cur, cur_w = "", 0
        for word in para.split():
            # Oversized word: flush the line, then hard-break the word itself.
            if _str_width(word) > width:
                if cur:
                    lines.append(cur)
                    cur, cur_w = "", 0
                pieces = _break_token(word, width)
                lines.extend(pieces[:-1])
                cur, cur_w = pieces[-1], _str_width(pieces[-1])
                continue
            add = _str_width(word) + (1 if cur else 0)
            if cur_w + add > width:
                lines.append(cur)
                cur, cur_w = word, _str_width(word)
            else:
                cur = f"{cur} {word}" if cur else word
                cur_w += add
        if cur:
            lines.append(cur)
    return "\n".join(lines)


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


def _report(scenario: Scenario, story: str):
    """Run the deterministic validators; return the ValidationReport (has .hard_pass)."""
    from islm.config import DEFAULT_THRESHOLDS as T
    from islm.validators import validate_story
    from islm.vocab.lemmatize import get_analyzer

    analyzer = get_analyzer(scenario.language)
    targets = {t.lower() for t in scenario.target_words}
    # Grade against the palette the model was TOLD it could use — the promise to the learner,
    # and the honest test of whether the story is actually i+1 for that learner.
    return validate_story(
        story, scenario.known_set(), targets, analyzer, T, language=scenario.language
    )


def score(scenario: Scenario, story: str) -> bool:
    """Grade the story with the same deterministic validators the eval uses, and spell out exactly
    what failed — which words are OOV, which sentences add >1 new word, which targets under-recur.
    Returns the hard-pass boolean."""
    from islm.config import DEFAULT_THRESHOLDS as T

    rep = _report(scenario, story)
    c, onew, rec = rep.coverage, rep.one_new_word, rep.recurrence

    print("SCORE (deterministic i+1 checks):")
    print(f"  HARD PASS: {'YES' if rep.hard_pass else 'NO'}")

    # 1. OOV / coverage
    ok = c.oov_rate <= T.max_oov_rate and c.coverage >= T.min_coverage
    print(
        f"\n  [{'OK ' if ok else 'BAD'}] vocabulary — OOV {c.oov_rate:.1%} "
        f"(limit {T.max_oov_rate:.0%}), coverage {c.coverage:.1%} (need {T.min_coverage:.0%})"
    )
    if c.oov_words:
        print(f"        {c.oov} out-of-vocab word(s) the learner was NOT given:")
        print(f"        {', '.join(c.oov_words)}")

    # 2. <=1 new word per sentence
    print(
        f"\n  [{'OK ' if onew.passed else 'BAD'}] pacing — at most "
        f"{T.max_new_words_per_sentence} new word per sentence (max seen: {onew.max_new_words})"
    )
    if not onew.passed:
        for s in onew.per_sentence:
            if s.index in onew.violations:
                words = ", ".join(s.new_words)
                print(f"        sentence {s.index + 1} adds {len(s.new_words)} new words: {words}")

    # 3. recurrence >=3x
    print(
        f"\n  [{'OK ' if rec.passed else 'BAD'}] recurrence — each target appears "
        f">= {rec.min_required}x"
    )
    print(f"        counts: {dict(rec.counts)}")
    if rec.below:
        under = ", ".join(f"'{w}' only {rec.counts[w]}x" for w in rec.below)
        print(f"        UNDER {rec.min_required}x: {under}")
    if rec.absent:
        print(f"        MISSING entirely: {', '.join(rec.absent)}")
    return rep.hard_pass


def _metrics(scenario: Scenario, story: str) -> dict:
    """Pull the headline numbers out of a validation report for the base-vs-tuned table."""
    rep = _report(scenario, story)
    c, onew, rec = rep.coverage, rep.one_new_word, rep.recurrence
    # below/absent overlap (an absent target is also "below"), so union them for a distinct count.
    under = set(rec.below) | set(rec.absent)
    return {
        "hard_pass": rep.hard_pass,
        "oov_rate": c.oov_rate,
        "coverage": c.coverage,
        "max_new": onew.max_new_words,
        "targets_under": len(under),
        "n_targets": len(scenario.target_words),
    }


def _fmt_delta(base: float, tuned: float, unit: str = "pts", scale: float = 100.0,
               lower_is_better: bool = True) -> str:
    """Signed change with an arrow marking whether it moved the right way."""
    d = (tuned - base) * scale
    good = (d < 0) if lower_is_better else (d > 0)
    arrow = "improved" if (good and abs(d) > 1e-9) else ("worse" if abs(d) > 1e-9 else "same")
    return f"{d:+.1f} {unit}  ({arrow})"


def compare(mode: str, base_path: str, adapter: str, no_think: bool, seed: int | None,
            max_new_tokens: int, temperature: float) -> None:
    """Run BASE and TUNED on the SAME scenario and show what the fine-tune improved. Prints both
    stories, then a side-by-side table of the i+1 metrics with the base->tuned change on each —
    so the demo shows the gain over base, not just a pass/fail stamp."""
    import torch

    from islm.eval.generators import HFGenerator

    seed = seed if seed is not None else random.randint(0, 10_000)
    scenario = build_scenario(mode, seed)

    on_gpu = torch.cuda.is_available()
    kw = dict(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        device_map="auto" if on_gpu else None,
        load_in_4bit=on_gpu,
        chat_kwargs={"enable_thinking": False} if no_think else None,
    )
    print(f"\nloading base ({base_path}) and tuned (+{adapter})...", flush=True)
    base_gen = HFGenerator(base_path, None, **kw)
    tuned_gen = HFGenerator(base_path, adapter, **kw)

    base_story = base_gen(scenario)
    tuned_story = tuned_gen(scenario)

    print("-" * 70)
    print("BASE (no adapter):\n")
    print(_wrap(base_story))
    print("-" * 70)
    print("TUNED (+ adapter):\n")
    print(_wrap(tuned_story))
    print("-" * 70)

    b, t = _metrics(scenario, base_story), _metrics(scenario, tuned_story)
    print("\nIMPROVEMENT OVER BASE (same scenario, base -> tuned):\n")
    hp = f"{'PASS' if b['hard_pass'] else 'FAIL'} -> {'PASS' if t['hard_pass'] else 'FAIL'}"
    hp_tag = "fixed" if (t["hard_pass"] and not b["hard_pass"]) else (
        "held" if t["hard_pass"] else "still failing")
    rows = [
        ("hard pass", hp, hp_tag),
        (
            "OOV rate (limit 2%)",
            f"{b['oov_rate']:.1%} -> {t['oov_rate']:.1%}",
            _fmt_delta(b["oov_rate"], t["oov_rate"]),
        ),
        (
            "coverage (need 98%)",
            f"{b['coverage']:.1%} -> {t['coverage']:.1%}",
            _fmt_delta(b["coverage"], t["coverage"], lower_is_better=False),
        ),
        (
            "max new words/sentence (<=1)",
            f"{b['max_new']} -> {t['max_new']}",
            _fmt_delta(b["max_new"], t["max_new"], unit="", scale=1.0),
        ),
        (
            "targets under-recurring",
            f"{b['targets_under']}/{b['n_targets']} -> {t['targets_under']}/{t['n_targets']}",
            _fmt_delta(b["targets_under"], t["targets_under"], unit="", scale=1.0),
        ),
    ]
    for label, change, delta in rows:
        print(f"  {label:<30} {change:<18} {delta}")
    print(f"\n(seed={seed} — pass --seed {seed} to reproduce this exact selection)")


def find_passing(mode: str, base_path: str, adapter: str, n: int, no_think: bool,
                 stop_on_first: bool = True) -> None:
    """Search seeds for the demo-ideal case: BASE fails the i+1 spec, TUNED passes it. Loads both
    models once and tries up to `n` seeds. Prints the winning seeds AND writes the first one to
    `.demo_seed_<mode>` so the demo notebook can pick it up automatically (no manual paste).
    Run this in Colab (where the adapter + GPU live). With `stop_on_first`, it returns as soon as it
    finds one ideal seed (fast); otherwise it scans all `n`.
    """
    import torch

    from islm.eval.generators import HFGenerator

    on_gpu = torch.cuda.is_available()
    kw = dict(
        max_new_tokens=320,
        temperature=0.0,
        device_map="auto" if on_gpu else None,
        load_in_4bit=on_gpu,
        chat_kwargs={"enable_thinking": False} if no_think else None,
    )
    print(f"loading base ({base_path}) and tuned (+{adapter})...", flush=True)
    base_gen = HFGenerator(base_path, None, **kw)
    tuned_gen = HFGenerator(base_path, adapter, **kw)

    hits = []
    for seed in range(n):
        sc = build_scenario(mode, seed)  # same scenario for both models
        base_pass = _report(sc, base_gen(sc)).hard_pass
        tuned_pass = _report(sc, tuned_gen(sc)).hard_pass
        tag = "IDEAL (base FAIL, tuned PASS)" if (tuned_pass and not base_pass) else (
            "both pass" if tuned_pass else "tuned fail")
        print(f"  seed {seed}: base={'PASS' if base_pass else 'FAIL'} "
              f"tuned={'PASS' if tuned_pass else 'FAIL'}  {sc.target_words}  -> {tag}")
        if tuned_pass and not base_pass:
            hits.append(seed)
            if stop_on_first:
                break
    print(f"\n[{mode}] ideal demo seeds (base fails, tuned passes): {hits or 'NONE in range'}")
    if hits:
        Path(f".demo_seed_{mode}").write_text(str(hits[0]), encoding="utf-8")
        print(f"  wrote seed {hits[0]} -> .demo_seed_{mode}  (the demo reads this automatically)")
    else:
        print("  NONE found — adapter likely stale/untrained, or bump --find-passing.")


def main() -> None:
    p = argparse.ArgumentParser(description="Manually test a trained i+1 story model.")
    p.add_argument("--mode", required=True, choices=list(MODES), help="en | en-exam | zh | jp")
    p.add_argument("--base-path", required=True, help="HF model path/id for the base.")
    p.add_argument("--adapter", default=None, help="LoRA adapter path (omit to test the base).")
    p.add_argument("--seed", type=int, default=None, help="Fix the selection (default: random).")
    p.add_argument("--max-new-tokens", type=int, default=320)
    p.add_argument("--no-think", action="store_true", help="Disable thinking mode (e.g. Qwen3).")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument(
        "--find-passing", type=int, default=0, metavar="N",
        help="Search N seeds for a base-fails/tuned-passes case (needs --adapter).",
    )
    p.add_argument(
        "--compare", action="store_true",
        help="Run base AND tuned on the same scenario and show the improvement (needs --adapter).",
    )
    args = p.parse_args()

    if args.find_passing:
        if not args.adapter:
            sys.exit("--find-passing needs --adapter (compares base vs tuned)")
        find_passing(args.mode, args.base_path, args.adapter, args.find_passing, args.no_think)
        return

    if args.compare:
        if not args.adapter:
            sys.exit("--compare needs --adapter (compares base vs tuned)")
        compare(args.mode, args.base_path, args.adapter, args.no_think, args.seed,
                args.max_new_tokens, args.temperature)
        return

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
    print(_wrap(story))
    print("-" * 70)

    score(scenario, story)
    print(f"\n(seed={seed} — pass --seed {seed} to reproduce this exact selection)")


if __name__ == "__main__":
    main()
