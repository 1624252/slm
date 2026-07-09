---
name: train-islm
description: Run a full fine-tuning iteration for the i+1 Story SLM — regenerate/curate data, train a LoRA/QLoRA adapter, evaluate base-vs-tuned on held-out + exam + golden sets with all three criteria families (deterministic checks, 8-dim LLM judge, cloze), track results, and document hyperparameters + deltas. Use when the user asks to train, retrain, run a training iteration, tune hyperparameters, or "do a Day-N run" for this project.
---

# Train the i+1 Story SLM

A training iteration is not just `train.sft`. It is the full loop, and **every run must be
documented** so progress is comparable across runs and reproducible on the GPU later. Follow the
workflow below in order. Run long steps in the background and monitor them.

## The loop (always all these steps)

1. **Regenerate + curate data** (unless reusing an unchanged dataset)
2. **Train** an adapter with explicit hyperparameters
3. **Eval** base-vs-tuned, tracked, on all three surfaces: **held-out** (all langs), **exam**
   (GRE/SAT/ACT), and **golden** (`--golden`). The judge (8-dim rubric) + cloze run automatically
   when the key is set — so every run reports the three criteria families (see Steps 3 & 4).
4. **Document** the run: hyperparameters + **all criteria** (deterministic + judge + cloze),
   base→tuned, in `evals/RESULTS_LOG.md`
5. **Commit** (conventional-commit message; never commit keys)

All commands run from the repo root. Prefix with `PYTHONPATH=src` unless the package is
`pip install -e .`'d. On Windows use the Bash tool for these POSIX commands.

## Environment facts (don't rediscover these)

- Package is **not** installed; use `PYTHONPATH=src`.
- No **teacher** key for data generation → use the **authored seed** (`data/curated/seed`). But a
  **judge** key IS configured in `.env` (`OPENAI_API_KEY` + `JUDGE_MODEL=claude-...`), so the LLM
  judge + cloze run by default; pass `--no-judge` only for a quick deterministic-only check.
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
`--grad-accum`, `--max-steps`, `--qlora` (GPU only), `--merge` (GPU; fp16 merged model for deploy).
`--smoke` = tiny fast loop (plumbing only).

**Recipe (baked into `sft.py` defaults — matches the reference QLoRA Colab notebook).** Every run
already uses: 4-bit nf4 + double-quant (under `--qlora`), LoRA on all-linear, **cosine LR schedule
+ 3% warmup**, weight_decay 0.001, max_grad_norm 0.3, `paged_adamw_32bit` on GPU (`adamw_torch` on
CPU), gradient checkpointing on GPU. Don't re-implement these — they're the config defaults, so any
run through `islm.train.sft` follows the notebook recipe automatically. On Colab add `--qlora`
(and `--merge`); see `docs/COLAB_PLAN.md` and `docs/TRAINING.md`.

Timing: on CPU, optimizer_steps = `epochs * train_examples / (batch * grad_accum)`; ~90 s each.
5 epochs × 22 ≈ 110 steps ≈ 2.5–3 h. **Run in background and Monitor** the log for
`final train loss` / `saved adapter` / `Traceback|Error`. The run writes
`outputs/.../train_summary.json` with every hyperparameter + `optimizer_steps` + final loss — this
is the machine record; do not hand-transcribe it wrong.

## Step 3 — Eval (run in background; tracked) — held-out + exam + golden

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

Golden set (the every-commit correctness target — run the model ON it, all criteria):

```bash
PYTHONPATH=src python -m islm.eval.run --golden \
    --base-path HuggingFaceTB/SmolLM2-135M-Instruct \
    --tuned-path HuggingFaceTB/SmolLM2-135M-Instruct --tuned-adapter outputs/day3_lora_v<N> \
    --track --run-label golden-v<N> --dataset data/curated/seed --epochs 5 --out evals/golden_v<N>
```

**The eval reports THREE families of criteria, each with base and tuned values:**
1. **Deterministic** (no AI): `hard_pass_rate` (all checks at once), `mean_oov_rate` (out-of-vocab,
   lower better, gate ≤0.02), `one_new_word_pass_rate`, `recurrence_pass_rate`.
2. **LLM-judge rubric** (8 dims, 0–2): spec_adherence, robustness, task_quality, consistency,
   inferability, seductive_detail_control, **coherence**, **interestingness**. Runs automatically
   when `OPENAI_API_KEY` + `JUDGE_MODEL` are in `.env`; `--no-judge` to skip.
3. **Cloze inferability**: `mean_inferability` — can the target be recovered from context.

Pull per-language numbers from `results_*.json` (`base` vs `tuned` dicts). Run held-out + exam +
golden targets so all three surfaces are covered.

## Step 4 — Document (REQUIRED — do not skip)

Add a newest-first entry to `evals/RESULTS_LOG.md` under `## Runs`. Include:

- **Iterations & hyperparameters table** — copy from `train_summary.json`: base model, epochs,
  optimizer_steps, batch/grad-accum, lr, lora_r/alpha/dropout, max_seq_len, seed, device, final
  loss, and the scheduler/optim recipe fields.
- **Results table(s)** — per language + target (held-out, exam, golden): ALL criteria as
  `base→tuned` — the 4 deterministic checks, cloze inferability, and all 8 judge dims. Show the
  base-model scores explicitly (they're in each results JSON's `base` block), not just deltas.
- **Read** — 2–3 sentences: did it beat the prior best? any regressions (flag honestly, e.g. a
  coherence/interestingness drop)? win condition still PASS/FAIL and why.

See the existing entries in `evals/RESULTS_LOG.md` for the exact format. The leaderboard
(`evals/LEADERBOARD.md`) is regenerated automatically by `--track`; never hand-edit it.

Optional — push the run to LangSmith (augment tracking, needs `LANGSMITH_API_KEY` in `.env`):
```bash
python -m islm.eval.langsmith_sync results --results evals/day3_v<N>/results_en.json \
    --experiment day3-seed-lora-v<N>
```

## Step 5 — Commit

```bash
git add evals/day3_v<N> evals/day3_v<N>_exam evals/golden_v<N> \
    evals/runs.jsonl evals/LEADERBOARD.md evals/RESULTS_LOG.md
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
