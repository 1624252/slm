# Day 1 — Setup, research & brainlift (checkpoint)

Spec Day 1: *get the environment running inference, research the behavior, complete the brainlift.*
Checkpoint: *the base model runs and responds; the target behavior is known; the spiky POVs match
the target behavior.*

## Checklist

| Day-1 item | Status | Evidence |
| --- | --- | --- |
| Environment runs inference | Done | `torch` (CPU) + `transformers` installed; a base model loads and generates (below). |
| Base model runs and responds | Done | `HuggingFaceTB/SmolLM2-135M-Instruct` loaded in ~94s and answered: *"A cat, a gentle creature of the night, lounging on a warm bed…"* |
| Target behavior is known | Done | Behavior Spec in `PRD.md` §4 and `EVALUATION.md`. |
| Research the behavior | Done | `brainlift.md` — Krashen & Loschky (core), Nation, Webb, Waring, Hunt, Mayer, SRS-Stories, IFEval. |
| Complete the brainlift | Done | `brainlift.md` (purpose, scope, DOK 1–4, sources). |
| Spiky POVs match the behavior | Done | SPOV1 "trick them into learning"; SPOV2 "a constrained LM can out-tutor a human at i+1" — both describe the story generator. |
| Base evaluation (this session) | Done | `evals/baseline/results_en.{md,json}` — see below. |

## Base-model evaluation (the point of Day 1)

Ran the eval harness on the prompted base model over 8 held-out English scenarios (small curated
vocabulary), measuring the deterministic Behavior-Spec checks:

| Metric | Base (`SmolLM2-135M-Instruct`) |
| --- | --- |
| Hard-check pass rate | **0.000** (0/8 stories are spec-compliant) |
| OOV rate | **0.413** (41% of words are outside the learner's vocabulary) |
| ≤1 new word / sentence | **0.000** |
| Recurrence satisfied | 0.125 |

Every output failed on OOV, coverage, and pacing. Example (targets included `clue`, `key`):

> *"The journey of a lifetime, filled with moments of joy and sorrow, and the thrill of discovery.
> A lost key, hidden in the depths of a forgotten book… The journey was a journey of
> self-discovery… "* (repeats, adult vocabulary, ignores the vocabulary limit)

**This is the baseline the project exists to beat.** A prompted base model cannot hold the i+1
constraints — exactly the spec's litmus test that makes this a fine-tuning problem, not a
prompting one. Full artifact: `evals/baseline/results_en.md` (+ per-story detail in the JSON).

## Notes

- **Model choice:** the project's intended base is a small **Qwen3 (0.6B–4B)** (PRD §11). For a
  CPU-only, offline Day-1 baseline we used **SmolLM2-135M-Instruct** (a spec-listed small base)
  because HF throttles unauthenticated downloads (~0.3 MB/s → Qwen3-0.6B would take ~85 min). The
  same eval runs on the real base with one command once a GPU / HF token / API is available:

  ```bash
  python -m islm.eval.run --language en --base-path Qwen/Qwen3-4B-Instruct --no-think \
      --curated --judge-model <judge> --adversarial --out evals/baseline
  ```

- Install the local-inference extras with `pip install -e .[hf]`.
