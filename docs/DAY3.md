# Day 3 — v1 dataset & real numbers (midweek gate)

Spec Day 3: *generate and filter real data; first real training run; first base-vs-tuned eval.*
Checkpoint: **midweek gate — base-vs-tuned numbers are on the board.**

This run was done **locally, without Colab/GPU** and **without a teacher API key**, so it uses the
genuine spec-passing **authored seed** as the v1 dataset and a CPU LoRA run. The loop, the first
real numbers, and results tracking over time are all in place; scale (teacher corpus + a GPU
QLoRA run) is the next lever and needs the deferred cloud/API access (see "Deviations").

## Checklist

| Day-3 item | Status | Evidence |
| --- | --- | --- |
| Generate + filter **real** data | Done (offline) | Authored seed regenerated + curated: 29 authored → **28 kept** (22 train / 2 val / 4 test), `data/curated/seed`. Genuine spec-passing prose (not the Day-2 junk). |
| First **real** training run | Done (CPU) | Real LoRA on `SmolLM2-135M-Instruct` over the curated seed → `outputs/day3_lora` (final train loss **2.33**). Real data + real hyperparameters, not the `--smoke` junk loop. |
| First base-vs-tuned eval | Done | `evals/day3/results_en.md` — deterministic checks, base vs tuned, 8 held-out English scenarios. |
| **Numbers on the board** | Done | Table below + `evals/LEADERBOARD.md`. |
| Results tracked **over time** | Done | `islm.eval.track` + `--track`: append-only `evals/runs.jsonl` → regenerated `evals/LEADERBOARD.md` (Day-1 → Day-2 → Day-3 history). |

## The numbers (this is the checkpoint)

Base (`SmolLM2-135M-Instruct`, prompted) vs the same model + the Day-3 LoRA adapter, on 8 held-out
English scenarios (curated small vocab, temperature 0):

| Metric | Base | Tuned | Delta | Better |
| --- | --- | --- | --- | --- |
| Hard-check pass rate | 0.000 | 0.000 | +0.000 | — |
| OOV rate | 0.413 | **0.393** | **−0.020** | tuned |
| ≤1 new word/sentence | 0.000 | 0.000 | +0.000 | — |
| Recurrence satisfied | 0.125 | 0.125 | +0.000 | — |

Win condition (spec): **FAIL** — expected. A tiny 135M model given a few gradient steps on 22
short examples will not clear the spec. As in Day 2, the machinery **responds to real data**
(OOV 0.413 → 0.393, a larger drop than Day-2's junk-data 0.413 → 0.406). Getting a real *win*
needs the two things the local box can't provide today: a **teacher-generated corpus** and a
**GPU** for a real multi-epoch QLoRA run.

## Results tracking over time

`--track` records every eval to an append-only history and regenerates the leaderboard, so the
progression is visible in one place (`evals/LEADERBOARD.md`):

| When | Label | Hard-pass | OOV | Recurrence | Data |
| --- | --- | --- | --- | --- | --- |
| 2026-07-08 | day3-seed-lora | 0.000→0.000 | 0.413→**0.393** | 0.125→0.125 | curated seed (n=22) |
| 2026-07-07 | day2-smoke-lora | 0.000→0.000 | 0.413→0.406 | 0.125→0.250 | day2 junk (n=40) |
| 2026-07-07 | day1-baseline | 0.000 | 0.413 | 0.125 | — (prompted base) |

Cells are `base→tuned (delta)`. See `docs/EVALUATION.md` → "Tracking results over time".

## Deviations from the spec (local environment, flagged per project rules)

| Spec Day-3 intent | What we did locally | Why | Same command scales when… |
| --- | --- | --- | --- |
| Teacher-distilled corpus (hundreds–thousands) | Authored seed (28 curated, spec-passing) | No `.env`/teacher API key on this box | `pipeline --n 4000 --model <teacher>` with a key |
| QLoRA 4-bit on a GPU | Plain LoRA on CPU | No CUDA GPU (`torch+cpu`) | `train.sft --qlora` on a CUDA GPU |
| Full multi-epoch run | 8 steps (~0.36 epoch), seq 256 | This CPU does ~**62 s/optimizer-step**; a full 3-epoch/1024-token run projected to ~90 min | GPU makes epochs cheap; raise `--epochs`/`--max-seq-len` |

These are environment limits, not code changes: the spec itself says *"CPU is smoke-only; real
training needs a GPU"* (`docs/TRAINING.md`). Everything runs unchanged on GPU + a teacher key.

## Findings → Day 4 (fix in data, not hyperparameters)

- **The prompt embeds the entire `KNOWN_WORDS` list**, so a training record is **5k–12k tokens**
  while the target story (at the end) is a few hundred. Right-truncation (TRL's only mode) would
  drop the story, so training now **left-truncates** to keep the completion (`fit_to_end` in
  `train/sft.py`). The deeper data fix for Day 4: shrink/reorder the prompt so `TARGET_WORDS` +
  the instruction + the story all fit in a small window.
- **Scale the dataset** with a teacher model (the authored seed caps volume far below "thousands").
- **Run the real QLoRA** on a GPU once available for the first genuine base-vs-tuned *win*.

## Reproduce

```bash
# 1. Real v1 dataset (authored seed, no model needed) + second-pass curation:
python -m islm.datagen.seed --out data/generated/seed --language all
python -m islm.datagen.curate --in data/generated/seed --out data/curated/seed

# 2. Real LoRA on CPU (bounded for this box; drop --max-steps and raise --epochs on a GPU):
python -m islm.train.sft --data data/curated/seed \
    --base HuggingFaceTB/SmolLM2-135M-Instruct \
    --epochs 1 --max-steps 8 --grad-accum 1 --max-seq-len 256 --out outputs/day3_lora

# 3. Base-vs-tuned eval, tracked to the leaderboard:
python -m islm.eval.run --language en --curated \
    --base-path HuggingFaceTB/SmolLM2-135M-Instruct \
    --tuned-path HuggingFaceTB/SmolLM2-135M-Instruct --tuned-adapter outputs/day3_lora \
    --max-new-tokens 220 --track --run-label day3-seed-lora --dataset data/curated/seed \
    --out evals/day3
```

Verify: `python -m pytest` (all green, incl. `tests/test_track.py`); open `evals/day3/results_en.md`
and `evals/LEADERBOARD.md`.

Next (Day 4): resolve one failure mode **in data** (shrink the prompt so targets + story fit),
regenerate, retrain, and show the improvement move on the leaderboard.
