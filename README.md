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
| `docs/spec.md` | The one-week build brief this project is scoped against. |
| `docs/brainlift.md` | Research brainlift: theory, experts, insights, sources. |
| `src/` | Data-generation, validators, training, and inference code. |
| `data/` | Generated datasets (git-ignored; published to the HF Hub). |
| `evals/` | Evaluation harness and results (base vs. tuned). |

## Theoretical core

- **Stephen Krashen** — comprehensible input, `i+1`, the affective filter, compelling input,
  and narrow reading. The reason the story must sit just above the learner's level and never
  feel like a drill.
- **Lester Loschky** — comprehension is not acquisition. The reason felt fluency and measured
  learning are evaluated *separately*, with hard validators rather than vibes.

See `docs/brainlift.md` for the full evidence base and `docs/PRD.md` for scope, base model,
dataset design, and evaluation plan.

## Status

Planning / pre-build. The dataset and the evaluation harness are the primary deliverables and
are built before any training run.
