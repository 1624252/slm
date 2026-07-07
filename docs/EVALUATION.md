# Evaluation

How we measure whether the fine-tuned model actually beats the prompted base at the target
behavior. Per `spec.md`, the eval is **built before any training** — it is the make-or-break
piece. It implements the spec's minimum bar and Appendix A rubric (PRD 14–15).

## The Behavior Spec (what we grade against)

> For a given known-vocabulary list `K`, target-word set `T`, and theme, the model returns a short
> story in the target language such that: (1) every word is in `K ∪ T` (or a trivially inferable
> proper noun); (2) no sentence introduces more than one word from `T`; (3) each introduced target
> word's meaning is inferable from its sentence's context; (4) every target word in `T` recurs at
> least 3 times; and (5) the story is coherent and compelling, with any humor/surprise attached to
> a target word rather than an off-topic detail. It never states that it is teaching or labels the
> target words.

## What the eval tests

The spec's minimum bar is three things; we implement all three plus robustness.

| Spec requirement | Our implementation |
| --- | --- |
| A behavioral check for the specific failure the spec forbids | **Deterministic validators** (below) |
| An LLM-as-judge scoring outputs against the Behavior Spec | **Judge rubric** (Appendix A, below) |
| A base-vs-tuned comparison | **`islm.eval.run`** → results table with deltas |

### 1. Deterministic behavioral checks (`validators/`, the primary gate)

Run on every output; these are the failures the spec forbids. Targets from PRD 14.2.

**OOV** stands for **out-of-vocabulary**: a running word whose lemma (or surface form) is not in
the learner's known set `K` or the target set `T`. **OOV rate** = OOV words ÷ total words, and
**coverage** = 1 − OOV rate (the fraction of the story the learner can already understand). So
"OOV ≤ 1%" means at most 1 word in 100 is outside the learner's vocabulary.

| Check | What it verifies | Target (tuned) |
| --- | --- | --- |
| Coverage / OOV | fraction of words outside `K ∪ T` (out-of-vocabulary) | OOV ≤ 1%, coverage ≥ 98% |
| ≤1 new word / sentence | pacing of new vocabulary | ≤ 1 in ≥ 95% of stories |
| Recurrence | each target repeated for spaced repetition | each target ≥ 3× |
| Inferability (cloze) | new word guessable from context | ≥ 60% recovered |

Coverage matches on lemma **or** surface, and for CJK on character decomposition, so natural
Chinese/Japanese text isn't falsely flagged (see `docs/dataset-and-eval.md`).

### 2. LLM-as-judge rubric (spec Appendix A; each 0/1/2)

| Dimension | 0 | 1 | 2 |
| --- | --- | --- | --- |
| `spec_adherence` | violates the target behavior | partially follows | fully embodies the spec |
| `robustness` | breaks on messy/adversarial input | wobbles | holds under pressure |
| `task_quality` | wrong or useless | acceptable | genuinely good |
| `consistency` | differs across similar inputs | mostly stable | reliable every time |
| `inferability` (extra) | new word not guessable | — | clearly guessable |
| `seductive_detail_control` (extra) | off-target humor / announces lesson | — | humor carries the word |

The first four are the spec dimensions; the win condition is judged on them. The judge is a
**secondary** signal — human↔judge correlation is only moderate (SRS-Stories r≈0.46–0.56), so the
deterministic checks are primary and a human should spot-check (Loschky: comprehension ≠
acquisition; PRD 14.6).

### 3. Robustness / adversarial (spec Appendix A "Robustness", PRD 14.5)

`eval/adversarial.py` builds scenarios that stress the behavior: a **small** known vocabulary,
**multiple, harder** targets, and **jargon-tempting** themes. Robustness = the hard-check pass
rate on this set. A brittle model leaks OOV words here even if it looks fine on the clean set.

## Required outputs & win condition

`islm.eval.run` writes `evals/results/results_<lang>.{md,json}` with:

- Mean of each metric and each judge dimension, **base vs tuned, with deltas**, on the same
  held-out scenarios.
- The **robustness** table (adversarial hard-check pass, base vs tuned).
- A **win-condition verdict** — the spec's bar: *tuned beats base on Spec adherence AND
  Robustness*.
- An **error-analysis** section: which deterministic checks the tuned model still fails, as the
  starting point for the human paragraph ("is it a data problem?").

## How to run (once you have a trained model)

The model under test is any `StoryGenerator`, so swapping models needs no harness change.

```bash
# Offline smoke (mock for every role) — proves the harness end to end:
python -m islm.eval.run --mock --language en --adversarial

# API models (OpenAI-compatible):
python -m islm.eval.run --base-model qwen3-4b-instruct --tuned-model my-tuned \
    --judge-model gpt-5 --adversarial

# A local fine-tuned checkpoint (base + LoRA adapter) — the trained-model path:
python -m islm.eval.run \
    --base-path Qwen/Qwen3-4B-Instruct \
    --tuned-path Qwen/Qwen3-4B-Instruct --tuned-adapter outputs/lora \
    --judge-model gpt-5 --adversarial

# Add --guard to evaluate the deployed system (inference-time validate-and-rewrite), not just
# the raw model.
```

Held-out and adversarial scenarios live at `evals/scenarios/{heldout,adversarial}_<lang>.jsonl`
(committed for reproducibility; auto-created on first run). They are scenario-level distinct from
the training/seed data, so there is no leakage. Results go to `evals/results/` (git-ignored).

## Success criteria (PRD 15)

- Tuned **beats prompted base on Spec adherence and Robustness** (primary).
- OOV ≤ 1%, ≤1-new-word ≥ 95%, recurrence ≥ 90% — all higher than base.
- A reproducible results table + error-analysis paragraph.

## Automated tests for the eval harness

The eval logic itself is unit-tested (`python -m pytest`), so the numbers can be trusted:

| Test (`tests/`) | What it locks down |
| --- | --- |
| `test_eval.py::test_spec_dimensions_match_appendix_a` | Judge dimensions match spec Appendix A |
| `test_eval.py::test_judge_parses_fenced_json_and_clamps` | Judge JSON parsing + 0–2 clamping |
| `test_eval.py::test_judge_handles_garbage` | Malformed judge output → all zeros (no crash) |
| `test_eval.py::test_evaluate_and_aggregate_with_mock` | Harness runs; aggregate has hard + judge metrics |
| `test_eval.py::test_adversarial_targets_not_in_small_known` | Adversarial scenarios are well-formed |
| `test_eval.py::test_error_analysis_counts_failures` | Error analysis tallies the right failure tags |
| `test_eval.py::test_results_markdown_has_sections_and_verdict` | Report has all sections + win verdict |
| `test_eval.py::test_guard_recovers_a_bad_generation` | Inference guard rewrites a failing story |
| `test_validators.py`, `test_multilingual.py` | The deterministic checks the eval relies on |
