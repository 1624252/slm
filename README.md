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
| `docs/spec.md` | The one-week build brief this project is scoped against. |
| `docs/brainlift.md` | Research brainlift: theory, experts, insights, sources. |
| `src/islm/` | The package: `vocab`, `validators`, `llm`, `datagen`, `eval`. |
| `tests/` | Unit tests for the validators + an offline end-to-end smoke test. |
| `data/vocab/` | Bundled sample word list (real datasets land in `data/generated/`, git-ignored). |
| `evals/` | Held-out scenarios (`scenarios/`) and results (`results/`, git-ignored). |

## Getting started

```bash
python -m venv .venv && pip install -e .
python -m spacy download en_core_web_sm     # recommended for real data

# Build a dataset and run the eval fully offline (mock teacher, no API key):
python -m islm.datagen.pipeline --n 20 --mock --out data/generated
python -m islm.eval.run --mock

python -m pytest        # tests        (23 passing)
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

The dataset pipeline and evaluation harness are built and runnable end-to-end (offline via a
mock teacher), before any training — as the spec requires. Next up: real teacher generation, the
QLoRA fine-tune, and the base-vs-tuned results table.
