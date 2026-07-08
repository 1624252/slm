# Colab GPU plan — spending $10 well

## Why

The local CPU runs proved the loop and moved the eval (English OOV 0.41 → 0.17), but they can't
clear the spec: a 135M model on 22 hand-authored examples caps out well short of the OOV ≤ 0.02
gate. The PRD's actual target is **Qwen3-4B-Instruct fine-tuned with QLoRA on a single 24 GB GPU**
(PRD §11). That needs a GPU we don't have locally. We have **$10 of Colab credit** — enough for the
first real base-vs-tuned *win*, if spent deliberately. This plan says how.

## Budget math

$10 ≈ **100 Colab compute units** (pay-as-you-go, $9.99/100). Unit burn rate by GPU (confirm the
live rates in Colab → *Change runtime type*; these are the long-standing published figures):

| GPU | VRAM | ~units/hr | Hours on 100 units | Fit for us |
| --- | --- | --- | --- | --- |
| **L4** | 24 GB | ~4.8 | **~20 h** | **Best value** — fits Qwen3-4B QLoRA, most hours per dollar |
| A100 | 40 GB | ~11.8 | ~8.5 h | Fastest; use only for the final long run |
| T4 | 16 GB | ~1.8 | ~55 h | Cheapest, but 16 GB is tight for 4B; fine for 1.7B |
| V100 | 16 GB | ~4.9 | ~20 h | No advantage over L4 for us |

**Rule of thumb:** develop on **L4**, reserve **A100** only for the one final full run. Never leave
an idle GPU runtime connected — units burn on connection time, not just compute.

## Spend allocation (~100 units)

| Phase | GPU | Est. units | What |
| --- | --- | --- | --- |
| 0. Smoke on GPU | L4 | ~3 | Confirm QLoRA loads Qwen3-4B in 4-bit, one train step, one eval. Catch env bugs cheap. |
| 1. Teacher data gen | (API, not Colab) | 0 | Generate the corpus with a teacher key — CPU/API, no GPU. Do this *before* burning GPU units. |
| 2. Hyperparameter sweep | L4 | ~30 | 3–4 short QLoRA runs (1 epoch, subsets) to pick LR / rank / epochs. Eval each. |
| 3. Full training run | A100 | ~25 | One multi-epoch QLoRA run on the full corpus with the winning config. |
| 4. Eval + robustness | L4 | ~10 | All 3 languages + adversarial + exam set, tracked to the leaderboard. |
| — Reserve | — | ~30 | Buffer for a failed run / a second full attempt. **Do not plan to spend 100/100.** |

## Concrete steps

### Before touching Colab (local / API, free of GPU units)
1. **Generate the teacher corpus.** With a teacher API key set, scale past the 28-story seed:
   ```bash
   python -m islm.datagen.pipeline --n 2000 --language en --model <teacher> --out data/generated/en
   python -m islm.datagen.pipeline --n 800  --language zh --model <teacher> --out data/generated/zh
   python -m islm.datagen.pipeline --n 800  --language ja --model <teacher> --out data/generated/ja
   python -m islm.datagen.curate --in data/generated/en --out data/curated/en   # + zh, ja
   ```
   Curation hard-filters to spec-passing records. This is the deliverable; do it off-GPU.
2. **Download the full word lists** (`python -m islm.vocab.download --language all`) so scenarios
   use the real graded vocab, including the merged GRE/SAT/ACT exam tier.
3. Push the repo + curated data somewhere Colab can pull (GitHub, or mount Drive).

### On Colab (the notebook)
A `notebooks/train_colab.ipynb` should do, in order:
```python
# 1. Setup (uses the repo's own installer + the GPU-only extras)
!git clone <repo> && cd slm && pip install -e . && pip install bitsandbytes unsloth

# 2. Smoke (Phase 0) — cheap sanity before the real burn
!python -m islm.train.sft --data data/curated/seed --base Qwen/Qwen3-4B-Instruct \
    --qlora --smoke --out outputs/colab_smoke

# 3. Sweep (Phase 2) — short runs, vary one knob at a time, eval each
!python -m islm.train.sft --data data/curated/en --base Qwen/Qwen3-4B-Instruct --qlora \
    --epochs 1 --lr 2e-4 --lora-r 16 --max-seq-len 2048 --out outputs/sweep_lr2e4
# ... repeat for lr 1e-4, rank 32, then eval each with islm.eval.run --track

# 4. Full run (Phase 3) — switch runtime to A100, winning config, more epochs
!python -m islm.train.sft --data data/curated/en --base Qwen/Qwen3-4B-Instruct --qlora \
    --epochs 3 --lr <best> --lora-r <best> --max-seq-len 2048 --out outputs/qwen3_4b_qlora

# 5. Eval everything, tracked (Phase 4)
!python -m islm.eval.run --base-path Qwen/Qwen3-4B-Instruct \
    --tuned-path Qwen/Qwen3-4B-Instruct --tuned-adapter outputs/qwen3_4b_qlora \
    --adversarial --track --run-label colab-qwen3-4b --dataset data/curated/en \
    --out evals/colab
!python -m islm.eval.run --language en --scenarios evals/scenarios/heldout_exam_en.jsonl \
    --base-path Qwen/Qwen3-4B-Instruct --tuned-path Qwen/Qwen3-4B-Instruct \
    --tuned-adapter outputs/qwen3_4b_qlora --track --run-label colab-qwen3-4b-exam --out evals/colab
```
Everything above already exists in the repo — the CLI flags (`--qlora`, `--lr`, `--lora-r`,
`--lora-alpha`, `--max-seq-len`, `--track`) are wired. Nothing new to build; only the notebook.

### After Colab (local, free)
- Download the adapter + `train_summary.json` + `evals/colab/*` and the updated `runs.jsonl`.
- Record the run in `evals/RESULTS_LOG.md` (config + deltas) and commit.

## Unit-saving discipline
- **Fits in 24 GB:** Qwen3-4B in 4-bit QLoRA ≈ 6–8 GB weights + activations — comfortable on L4.
  If VRAM is tight, drop `--max-seq-len` before dropping the model.
- **Disconnect immediately** when a cell finishes; don't leave the runtime idle.
- **Checkpoint to Drive** so a disconnect mid-run isn't a total loss (`--out` on a mounted path).
- **Sweep small, run big once.** Most units die in careless full runs with an untested config.
- If $10 looks tight for Qwen3-4B, fall back to **Qwen3-1.7B** (fits a T4 → far more hours) and keep
  4B for the single A100 final run.

## What "success" looks like
The win condition (PRD 15): tuned **beats prompted base on Spec adherence AND Robustness**. On the
board that means the first non-trivial `hard_pass_rate` and a clear OOV drop across en/zh/ja *and*
the exam set — the numbers the CPU runs couldn't reach. Track every run so the CPU→GPU jump is
visible in `evals/LEADERBOARD.md`.
