# Training (QLoRA / LoRA SFT)

Supervised fine-tuning of a small open base model on the generated i+1 story dataset
(`src/islm/train`). Per the spec stack, the intended path is **QLoRA on a single GPU**; on CPU
or without bitsandbytes it falls back to plain **LoRA** (used for the Day-2 end-to-end smoke).

## Install

```bash
pip install -e .[train]          # torch, transformers, trl, peft, datasets, accelerate
pip install bitsandbytes          # only for 4-bit QLoRA on a CUDA GPU
```

## What it trains on

The JSONL chat records from `datagen`/`seed` — each record's `messages` (system rules + user
`(language, K, T, theme)` + assistant story) are rendered with the base model's chat template and
used as the SFT target. (Completion-only loss masking is a planned refinement; today it trains on
the full rendered text.)

## Run

```bash
# Real run (GPU, QLoRA 4-bit):
python -m islm.train.sft --data data/generated/en --base Qwen/Qwen3-4B-Instruct --qlora \
    --out outputs/lora

# End-to-end smoke (CPU, tiny model, a few steps) — proves the loop:
python -m islm.train.sft --data data/generated/day2_smoke \
    --base HuggingFaceTB/SmolLM2-135M-Instruct --smoke --out outputs/day2_lora
```

Key flags: `--epochs`, `--max-steps`, `--qlora` (needs a CUDA GPU), `--smoke` (tiny CPU settings).
LoRA rank/alpha/lr and sequence length live in `TrainConfig` (`src/islm/train/sft.py`). Adapters
are written to `--out` (git-ignored; publish to the HF Hub).

## Then evaluate the adapter (closes the loop)

```bash
python -m islm.eval.run --language en \
    --base-path HuggingFaceTB/SmolLM2-135M-Instruct \
    --tuned-path HuggingFaceTB/SmolLM2-135M-Instruct --tuned-adapter outputs/day2_lora \
    --curated --judge-model <judge> --adversarial --out evals/results
```

This is the full **generate → train → eval** loop. See `docs/EVALUATION.md` for the metrics.

## Notes

- **QLoRA specifics** (spec/PRD): 4-bit NF4, frozen base, LoRA on all linear layers, paged
  optimizer — set on GPU via `--qlora`. Fits a small Qwen3 on a single 24 GB card.
- **CPU is smoke-only.** Real training needs a GPU (Colab/Modal/RunPod, per the spec).
