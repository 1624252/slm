---
name: train-islm
description: Run a full fine-tuning iteration for the i+1 Story SLM — regenerate/curate data, train a LoRA/QLoRA adapter, evaluate base-vs-tuned on all languages plus the exam set, track results, and document hyperparameters + deltas. Use when the user asks to train, retrain, run a training iteration, tune hyperparameters, or "do a Day-N run" for this project.
---

# Train the i+1 Story SLM

A training iteration is not just `train.sft`. It is the full loop, and **every run must be
documented** so progress is comparable across runs and reproducible on the GPU later. Follow the
workflow below in order. Run long steps in the background and monitor them.

## The loop (always all six steps)

1. **Regenerate + curate data** (unless reusing an unchanged dataset)
2. **Train** an adapter with explicit hyperparameters
3. **Eval** base-vs-tuned on all shipped languages, tracked
4. **Eval on the exam set** (GRE/SAT/ACT targets), tracked
5. **Document** the run: hyperparameters + eval deltas in `evals/RESULTS_LOG.md`
6. **Commit** (conventional-commit message; never commit keys)

All commands run from the repo root. Prefix with `PYTHONPATH=src` unless the package is
`pip install -e .`'d. On Windows use the Bash tool for these POSIX commands.

## Environment facts (don't rediscover these)

- Package is **not** installed; use `PYTHONPATH=src`.
- No `.env`/teacher key locally → use the **authored seed** (`data/curated/seed`), not API generation.
- No CUDA GPU locally → plain **LoRA on CPU** (~90 s/optimizer-step). `--qlora` needs a GPU (Colab).
- Base model for local runs: `HuggingFaceTB/SmolLM2-135M-Instruct`. Intended GPU base: `Qwen/Qwen3-4B-Instruct` (see `docs/COLAB_PLAN.md`).
- Prompts embed `KNOWN_WORDS`; the seed scopes it to ~150 words so records fit a 768-token window. Keep `--max-seq-len` ≥ 768 locally.
- `data/curated/*` and `outputs/*` are git-ignored; `evals/day*/`, `runs.jsonl`, `LEADERBOARD.md`, `RESULTS_LOG.md` are committed.

## Step 1 — Data

```bash
PYTHONPATH=src python -m islm.datagen.seed  --out data/generated/seed --language all
PYTHONPATH=src python -m islm.datagen.curate --in data/generated/seed --out data/curated/seed
```
Confirm the counts (expect ~29 authored → 28 kept → 22 train). If a teacher key is available,
scale instead with `islm.datagen.pipeline --n <N> --model <teacher>` (see `docs/COLAB_PLAN.md`).

## Step 2 — Train (run in background; monitor)

Pick a **name** like `day3_lora_v<N>`. Change **one variable per iteration** so comparisons are clean.

```bash
PYTHONPATH=src python -m islm.train.sft --data data/curated/seed \
    --base HuggingFaceTB/SmolLM2-135M-Instruct \
    --epochs 5 --grad-accum 1 --max-seq-len 768 --lr 2e-4 --lora-r 32 --lora-alpha 64 \
    --out outputs/day3_lora_v<N>
```

Tunable knobs (all CLI flags): `--epochs`, `--lr`, `--lora-r`, `--lora-alpha`, `--max-seq-len`,
`--grad-accum`, `--max-steps`, `--qlora` (GPU only). `--smoke` = tiny fast loop (plumbing only).

Timing: on CPU, optimizer_steps = `epochs * train_examples / (batch * grad_accum)`; ~90 s each.
5 epochs × 22 ≈ 110 steps ≈ 2.5–3 h. **Run in background and Monitor** the log for
`final train loss` / `saved adapter` / `Traceback|Error`. The run writes
`outputs/.../train_summary.json` with every hyperparameter + `optimizer_steps` + final loss — this
is the machine record; do not hand-transcribe it wrong.

## Steps 3 & 4 — Eval (run in background; tracked)

All languages (writes `evals/day3_v<N>/results_{en,zh,ja}.{md,json}`, appends leaderboard rows):

```bash
PYTHONPATH=src python -m islm.eval.run --curated \
    --base-path HuggingFaceTB/SmolLM2-135M-Instruct \
    --tuned-path HuggingFaceTB/SmolLM2-135M-Instruct --tuned-adapter outputs/day3_lora_v<N> \
    --max-new-tokens 220 --track --run-label day3-seed-lora-v<N> --dataset data/curated/seed \
    --epochs 5 --notes "<one-line summary of the config change>" --out evals/day3_v<N>
```

Exam set (GRE/SAT/ACT targets the model never trained on — the hard generalization test):

```bash
PYTHONPATH=src python -m islm.eval.run --language en \
    --base-path HuggingFaceTB/SmolLM2-135M-Instruct \
    --tuned-path HuggingFaceTB/SmolLM2-135M-Instruct --tuned-adapter outputs/day3_lora_v<N> \
    --scenarios evals/scenarios/heldout_exam_en.jsonl --max-new-tokens 200 \
    --track --run-label day3-v<N>-exam --dataset data/curated/seed --epochs 5 --out evals/day3_v<N>_exam
```

Metrics (all deterministic, no AI grader): `hard_pass_rate` (all checks at once), `mean_oov_rate`
(out-of-vocab, lower better, gate ≤0.02), `one_new_word_pass_rate`, `recurrence_pass_rate`. Pull
per-language numbers from the `results_*.json` (`base` vs `tuned` dicts).

## Step 5 — Document (REQUIRED — do not skip)

Add a newest-first entry to `evals/RESULTS_LOG.md` under `## Runs`. Include:

- **Iterations & hyperparameters table** — copy from `train_summary.json`: base model, epochs,
  optimizer_steps, batch/grad-accum, lr, lora_r/alpha/dropout, max_seq_len, seed, device, final loss.
  Note what changed vs the previous run and why.
- **Results table** — per language (+ exam): OOV, hard-pass, ≤1-new, recurrence, each `base→tuned`,
  with targets in the header (`OOV (→ ≤0.02)`, `Hard-pass (→ 1.000)`, …).
- **Read** — 2–3 sentences: did it beat the prior best? any regressions (flag honestly)? win
  condition still PASS/FAIL and why.

See the existing entries in `evals/RESULTS_LOG.md` for the exact format. The leaderboard
(`evals/LEADERBOARD.md`) is regenerated automatically by `--track`; never hand-edit it.

Optional — push the run to LangSmith (augment tracking, needs `LANGSMITH_API_KEY` in `.env`):
```bash
python -m islm.eval.langsmith_sync results --results evals/day3_v<N>/results_en.json \
    --experiment day3-seed-lora-v<N>
```

## Step 6 — Commit

```bash
git add evals/day3_v<N> evals/day3_v<N>_exam evals/runs.jsonl evals/LEADERBOARD.md evals/RESULTS_LOG.md
git commit -m "eval(day3): v<N> (<one-line change>) — <headline delta>"
```
`outputs/` is git-ignored, so the adapter itself isn't committed — the numbers and `RESULTS_LOG`
entry are the durable record. End the commit body with the project's Co-Authored-By trailer.

## Iterating until improvement

- Compare the new run's numbers to the current best (top of `RESULTS_LOG.md` / leaderboard).
- **Improved** (lower OOV, more hard-passes, no bad regressions) → keep it, it's the new best.
- **Not improved / regressed** → change one knob and rerun. Cheap high-leverage knobs on this tiny
  dataset: more `--epochs`, higher `--lora-r`/`--lora-alpha` (capacity), lower `--lr` (if a language
  regressed — usually over-fit). Log every attempt, including the ones that didn't help.
- Local ceiling: a 135M CPU model on 22 examples **will not** clear the OOV ≤ 0.02 gate. Real wins
  need the GPU run in `docs/COLAB_PLAN.md`. Don't burn hours chasing a gate the setup can't reach.

## Regression gate (run before/after every training change)

The golden set (Layer 1) must stay green — it's the "all must pass" correctness gate:
```bash
python -m pytest tests/test_golden.py -q
```
If it fails, stop: a validator or the data regressed, not the model.

## Related docs

- `docs/TRAINING.md` — training flags and the truncation rationale.
- `docs/EVALUATION.md` — every metric, the rubric, and the six eval layers.
- `docs/GOLDEN_SET.md` / `docs/ERROR_ANALYSIS.md` — the golden gate and failure taxonomy.
- `docs/COLAB_PLAN.md` — the GPU (Qwen3-4B QLoRA) plan and budget.
- `evals/RESULTS_LOG.md` — the run history this skill appends to.
