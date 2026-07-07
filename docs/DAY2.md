# Day 2 — Spec, eval & smoke test (checkpoint)

Spec Day 2: *write the Behavior Spec; build the eval harness and data-gen pipeline; run 50 junk
examples.* Checkpoint: **the full loop (generate → train → eval) runs end to end.**

## Checklist

| Day-2 item | Status | Evidence |
| --- | --- | --- |
| Write the Behavior Spec | Done | `PRD.md` §4 and `EVALUATION.md` (falsifiable pass/fail rubric). |
| Build the eval harness | Done | `src/islm/eval/` — validators + LLM judge + robustness + base-vs-tuned report. |
| Build the data-gen pipeline | Done | `src/islm/datagen/` — scenarios → generate → validate → rewrite → curate. |
| Run 50 junk examples | Done | `python -m islm.datagen.pipeline --n 50 --mock` → 50 kept (40/5/5 split). |
| Build the training step | Done | `src/islm/train/sft.py` — QLoRA (GPU) / LoRA (CPU) SFT. |
| **Full loop runs end to end** | Done | generate → train → eval, below. |

## The end-to-end smoke (this is the checkpoint)

Ran the whole loop on CPU with junk data to prove the plumbing (not to produce a good model):

1. **Generate** — 50 junk examples via the mock teacher → `data/generated/day2_smoke/` (40 train).
2. **Train** — LoRA on `SmolLM2-135M-Instruct` (`--smoke`, 3 steps) → adapter at
   `outputs/day2_lora` (train loss ≈ 3.48; TRL `SFTTrainer` + PEFT).
3. **Eval** — base vs. the LoRA-tuned adapter → `evals/day2/results_en.md`.

The loop completes without error. As expected for junk data + 3 steps, quality is unchanged
(both fail the spec), though even this nudged **recurrence 0.125 → 0.250** and **OOV 0.413 →
0.406** — the machinery responds to data. Real gains come from real data + a real (GPU) train run.

## Reproduce

```bash
python -m islm.datagen.pipeline --n 50 --language en --mock --out data/generated/day2_smoke
python -m islm.train.sft --data data/generated/day2_smoke \
    --base HuggingFaceTB/SmolLM2-135M-Instruct --smoke --out outputs/day2_lora
python -m islm.eval.run --language en --base-path HuggingFaceTB/SmolLM2-135M-Instruct \
    --tuned-path HuggingFaceTB/SmolLM2-135M-Instruct --tuned-adapter outputs/day2_lora \
    --curated --out evals/day2
```

Next (Day 3): generate + filter **real** data (teacher model), run a real QLoRA train on a GPU,
and put the first real base-vs-tuned numbers on the board.
