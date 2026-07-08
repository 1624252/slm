# Day 3 — v1 dataset & real numbers (midweek gate)

Spec Day 3: *generate and filter real data; first real training run; first base-vs-tuned eval.*
Checkpoint: **midweek gate — base-vs-tuned numbers are on the board.**

Run **locally, without Colab/GPU** and **without a teacher API key**, so it uses the genuine
spec-passing **authored seed** as the v1 dataset and a CPU LoRA run. The loop, the first real
numbers, and results tracking over time are all in place; scale (teacher corpus + a GPU QLoRA run)
is the next lever and needs the deferred cloud/API access (see "Deviations").

## Checklist

| Day-3 item | Status | Evidence |
| --- | --- | --- |
| Generate + filter **real** data | Done (offline) | Authored seed, compact prompts: 29 authored → **28 kept** (22 train / 2 val / 4 test), `data/curated/seed`. Genuine spec-passing prose. |
| First **real** training run | Done (CPU) | LoRA on `SmolLM2-135M-Instruct`, 3 epochs / 66 steps over the curated seed → `outputs/day3_lora` (final train loss **1.63**). |
| First base-vs-tuned eval | Done | `evals/day3/results_{en,zh,ja}.md` — deterministic checks, base vs tuned, 8 held-out scenarios each. |
| **Numbers on the board** | Done | Table below + `evals/LEADERBOARD.md`. |
| Results tracked **over time** | Done | `--track` → append-only `evals/runs.jsonl` → regenerated `evals/LEADERBOARD.md`; narrative in `evals/RESULTS_LOG.md`. |

## The numbers (this is the checkpoint)

Base (`SmolLM2-135M-Instruct`, prompted) vs the same model + the Day-3 LoRA adapter, 8 held-out
scenarios per language (curated small vocab, temperature 0):

Targets in parentheses are the Behavior-Spec thresholds (`config.Thresholds`).

| Lang | OOV (→ ≤0.02) | Hard-pass (→ 1.000) | ≤1-new (→ 1.000) | Recurrence (→ 1.000) |
| --- | --- | --- | --- | --- |
| en | 0.413 → **0.170** (−0.243) | 0.000 → 0.000 | 0.000 → 0.000 | 0.125 → 0.125 |
| zh | 0.856 → **0.439** (−0.417) | 0.000 → **0.250** | 0.000 → **0.250** | 0.125 → **0.375** |
| ja | 1.000 → **0.874** (−0.126) | 0.000 → 0.000 | 0.000 → **0.125** | 0.000 → 0.000 |

Win condition (spec): **FAIL** — expected. A 135M model given CPU LoRA on 22 short examples will
not clear the OOV ≤ 0.02 gate. But the eval **improved on every language**: OOV fell across the
board, and **Chinese posted the project's first non-zero hard-pass (0.25)**. A real *win* needs the
two things the local box can't provide: a **teacher-generated corpus** and a **GPU** for a real
multi-epoch QLoRA run.

## What made this work (vs. the first, deleted Day-3 attempt)

The earlier run's tuned model never moved off `hard_pass` 0 and barely touched OOV (−0.020). The
cause was a **data bug**: each training record embedded the full **2291-word** baseline as
`KNOWN_WORDS`, so records were **5k–12k tokens**. Training left-truncates to keep the story (at the
end), which dropped `TARGET_WORDS` and the rules (before the word list) — the model never saw the
task, and the train prompt didn't match the 146-word eval prompt. `seed.py` now scopes each
record's `KNOWN_WORDS` to the curated baseline + the story's own words (~150), so records are
**539–704 tokens** and fit a 768-token window uncut. Full write-up: `evals/RESULTS_LOG.md`.

## Deviations from the spec (local environment, flagged per project rules)

| Spec Day-3 intent | What we did locally | Why | Same command scales when… |
| --- | --- | --- | --- |
| Teacher-distilled corpus (hundreds–thousands) | Authored seed (28 curated, spec-passing) | No `.env`/teacher API key on this box | `pipeline --n 4000 --model <teacher>` with a key |
| QLoRA 4-bit on a GPU | Plain LoRA on CPU | No CUDA GPU (`torch+cpu`) | `train.sft --qlora` on a CUDA GPU |
| Full multi-epoch run | 3 epochs / 66 steps, seq 768 | This CPU does ~90 s/step | GPU makes epochs cheap; raise `--epochs`/`--max-seq-len` |

These are environment limits, not code changes; everything runs unchanged on GPU + a teacher key.

## Reproduce

```bash
# 1. v1 dataset (authored seed, compact prompts, no model needed) + second-pass curation:
python -m islm.datagen.seed --out data/generated/seed --language all
python -m islm.datagen.curate --in data/generated/seed --out data/curated/seed

# 2. LoRA on CPU (3 epochs; add --qlora and raise --max-seq-len on a GPU):
python -m islm.train.sft --data data/curated/seed \
    --base HuggingFaceTB/SmolLM2-135M-Instruct \
    --epochs 3 --grad-accum 1 --max-seq-len 768 --out outputs/day3_lora

# 3. Base-vs-tuned eval on all three languages, tracked to the leaderboard:
python -m islm.eval.run --curated \
    --base-path HuggingFaceTB/SmolLM2-135M-Instruct \
    --tuned-path HuggingFaceTB/SmolLM2-135M-Instruct --tuned-adapter outputs/day3_lora \
    --max-new-tokens 220 --track --run-label day3-seed-lora-v2 --dataset data/curated/seed \
    --epochs 3 --out evals/day3
```

Verify: `python -m pytest` (all green); open `evals/day3/results_en.md`, `evals/LEADERBOARD.md`,
and `evals/RESULTS_LOG.md`.

Next: the **Colab GPU** run — teacher corpus + QLoRA on a larger base for the first genuine
base-vs-tuned *win*; and more Japanese seed data (ja is the current laggard at OOV 0.87).
