# i+1 Story SLM

A small, fine-tuned language model that writes **comprehensible-input** language-learning
stories. Every story stays inside a learner's known vocabulary, introduces **at most one new
word per sentence**, makes each new word inferable from context, recycles target words for
spaced repetition, and stays compelling — without announcing that it is a lesson.

The thesis is *behavior from data*: a well-prompted base model cannot reliably hold all of
these lexical, pedagogical, and formatting constraints at once. We instill that reliability by
fine-tuning (QLoRA) a small open model on data we generate and hard-filter, backed by
deterministic validators.

## Repository layout

| Path | Contents |
| --- | --- |
| `docs/PRD.md` | Product Requirements Document — start here. |
| `docs/dataset-and-eval.md` | How the dataset and evaluation work + how to run them. |
| `docs/EVALUATION.md` | Behavior Spec, all eval checks + rubric, how to eval a trained model. |
| `docs/TRAINING.md` | QLoRA/LoRA fine-tuning: install, data, run, then eval the adapter. |
| `docs/COLAB_PLAN.md` | How to spend the $10 Colab credit: Qwen3-4B QLoRA, budget, steps. |
| `docs/DAY1.md` | Day-1 checkpoint: env/inference, brainlift, and the base-model baseline. |
| `docs/DAY3.md` | Day-3 checkpoint: real v1 dataset, first real (CPU) train, first base-vs-tuned numbers. |
| `evals/RESULTS_LOG.md` | Narrative results log: what changed each run, why, and what the eval did. |
| `docs/DATA_CARD.md` | The data recipe: generation, two-pass curation, counts, licenses. |
| `docs/design-constraints.md` | Constraint-first + representation analysis of the design. |
| `docs/spec.md` | The one-week build brief this project is scoped against. |
| `docs/brainlift.md` | Research brainlift: theory, experts, insights, sources. |
| `src/islm/` | The package: `vocab`, `validators`, `llm`, `datagen`, `eval`, `train`. |
| `tests/` | Unit tests for the validators + an offline end-to-end smoke test. |
| `data/vocab/<lang>/` | Per-language baseline + advanced word lists (real datasets land in `data/generated/`, git-ignored). |
| `evals/` | Held-out scenarios (`scenarios/`), per-run results, and the cross-run `LEADERBOARD.md` (+ `runs.jsonl` history). |

## Getting started

Shipped languages: **English, Chinese, Japanese** (`--language en|zh|ja`); the pipeline is
language-agnostic and any other language falls back to frequency bands + a generic tokenizer.

```bash
python -m venv .venv && pip install -e .    # installs spaCy, jieba, fugashi, wordfreq, ...
python -m spacy download en_core_web_sm     # recommended for English

# Build a dataset and run the eval fully offline (mock teacher, no API key):
python -m islm.datagen.pipeline --n 20 --language zh --mock --out data/generated/zh
python -m islm.eval.run --mock            # evals all shipped languages (en, zh, ja)

python -m pytest        # tests (62 passing, incl. zh/ja + tracker)
ruff check src tests    # lint
```

For real generation, copy `.env.example` to `.env` and add your key (never commit it). See
`docs/dataset-and-eval.md` for the full workflow, CLI flags, and the record schema.

## Theoretical core

- **Stephen Krashen** — comprehensible input, `i+1`, the affective filter, compelling input,
  and narrow reading. The reason the story must sit just above the learner's level and never
  feel like a drill.
- **Lester Loschky** — comprehension is not acquisition. The reason felt fluency and measured
  learning are evaluated *separately*, with hard validators rather than vibes.

See `docs/brainlift.md` for the full evidence base and `docs/PRD.md` for scope, base model,
dataset design, and evaluation plan.

## Status

The full loop — **generate → train → eval** — runs end to end, and the first **base-vs-tuned
numbers are on the board** (Day-3 midweek gate). Day 3 was run locally on CPU over the genuine
authored seed (`data/curated/seed`), with results tracked across runs in `evals/LEADERBOARD.md`
and narrated in `evals/RESULTS_LOG.md`. The tuned model improves the eval on all three languages
(e.g. English OOV 0.41 → 0.17; Chinese posts the first non-zero hard-pass, 0.25). A full spec
*win* needs the deferred cloud pieces: a **teacher-generated corpus** and a **GPU QLoRA** run. See
`docs/DAY3.md`.
