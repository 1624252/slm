"""Supervised fine-tuning (QLoRA / LoRA) for the i+1 story model.

QLoRA (4-bit) on a single GPU is the intended path (spec stack: Unsloth or TRL/PEFT). On CPU or
without bitsandbytes it falls back to plain LoRA — enough to run the Day-2 end-to-end smoke. It
trains on the JSONL chat records produced by `datagen`/`seed` (the `messages` field).

    # Real run (GPU):
    python -m islm.train.sft --data data/generated/en --base Qwen/Qwen3-4B-Instruct \
        --qlora --out outputs/lora

    # End-to-end smoke (CPU, tiny model, a few steps):
    python -m islm.train.sft --data data/curated/seed \
        --base HuggingFaceTB/SmolLM2-135M-Instruct --smoke --out outputs/smoke_lora

Adapters are written to `--out` (git-ignored; publish to the HF Hub). Evaluate with:
    python -m islm.eval.run --base-path <base> --tuned-path <base> --tuned-adapter <out> ...
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

# Tiny overrides for a fast CPU smoke (loop-runs, not a real train).
SMOKE = {
    "epochs": 1.0,
    "max_steps": 3,
    "per_device_batch_size": 1,
    "grad_accum": 1,
    "max_seq_len": 256,
    "lora_r": 8,
    "lora_alpha": 16,
}


@dataclass
class TrainConfig:
    base_model: str
    data_dir: Path
    output_dir: Path
    epochs: float = 1.0
    max_steps: int = -1
    learning_rate: float = 2e-4
    per_device_batch_size: int = 1
    grad_accum: int = 8
    max_seq_len: int = 1024
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    qlora: bool = False
    seed: int = 0


def read_records(data_dir: Path) -> list[dict]:
    """Read all `train.jsonl` chat records from a dataset directory."""
    path = Path(data_dir) / "train.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def render_plain(messages: list[dict]) -> str:
    """Fallback rendering (no tokenizer): `role: content` blocks. Used for tests/debug."""
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def fit_to_end(text: str, tokenizer, max_tokens: int) -> str:
    """Left-truncate so the assistant story (at the end) survives the context window.

    The prompt embeds the full KNOWN_WORDS list, so a record can be >10k tokens. TRL only
    truncates from the start (`keep_start`), which would drop the completion we actually train
    on. So keep the last `max_tokens` tokens here instead, before TRL ever sees the text.
    """
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[-max_tokens:])


def load_texts(data_dir: Path, tokenizer=None, max_tokens: int | None = None) -> list[str]:
    """Render each chat record to one training string via the tokenizer's chat template.

    With a tokenizer and `max_tokens`, over-long records are left-truncated (see `fit_to_end`)
    so the target story is not truncated away.
    """
    texts = []
    for rec in read_records(data_dir):
        messages = rec["messages"]
        if tokenizer is not None:
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            if max_tokens is not None:
                text = fit_to_end(text, tokenizer, max_tokens)
            texts.append(text)
        else:
            texts.append(render_plain(messages))
    return texts


def train(config: TrainConfig) -> Path:
    """Fine-tune and save a LoRA adapter to `config.output_dir`. Returns the output path."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    on_gpu = torch.cuda.is_available()
    model_kwargs: dict = {}
    if config.qlora and on_gpu:  # 4-bit QLoRA needs a CUDA GPU (bitsandbytes)
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(config.base_model, **model_kwargs)

    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    texts = load_texts(config.data_dir, tokenizer, max_tokens=config.max_seq_len)
    dataset = Dataset.from_dict({"text": texts})
    args = SFTConfig(
        output_dir=str(config.output_dir),
        num_train_epochs=config.epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.grad_accum,
        learning_rate=config.learning_rate,
        max_length=config.max_seq_len,
        dataset_text_field="text",
        packing=False,
        logging_steps=1,
        report_to=[],
        seed=config.seed,
        bf16=on_gpu,
    )
    # Pass our tokenizer so training uses the same rendering/truncation we fit the texts to.
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    train_loss = (
        trainer.state.log_history[-1].get("train_loss") if trainer.state.log_history else None
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(config.output_dir))
    tokenizer.save_pretrained(str(config.output_dir))

    # Full, human-readable record of what produced this adapter (for the run log / DAY docs).
    # `optimizer_steps` is the actual iteration count trainer ran (epochs*examples/(batch*accum),
    # or max_steps if capped) — the "number of iterations" for reproducibility.
    summary = {
        "base_model": config.base_model,
        "data_dir": str(config.data_dir),
        "train_examples": len(texts),
        "epochs": config.epochs,
        "max_steps": config.max_steps,
        "optimizer_steps": trainer.state.global_step,
        "per_device_batch_size": config.per_device_batch_size,
        "grad_accum": config.grad_accum,
        "max_seq_len": config.max_seq_len,
        "learning_rate": config.learning_rate,
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "seed": config.seed,
        "qlora": config.qlora and on_gpu,
        "device": "cuda" if on_gpu else "cpu",
        "final_train_loss": round(train_loss, 4) if train_loss is not None else None,
    }
    with open(config.output_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"final train loss: {summary['final_train_loss']}")
    return config.output_dir


def main() -> None:
    p = argparse.ArgumentParser(description="SFT (QLoRA/LoRA) for the i+1 story model.")
    p.add_argument("--data", type=Path, required=True, help="Dataset dir with train.jsonl.")
    p.add_argument("--base", required=True, help="Base model (HF path or hub id).")
    p.add_argument("--out", type=Path, default=Path("outputs/lora"))
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--max-seq-len", type=int, default=1024, help="Context window (left-truncated).")
    p.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps.")
    p.add_argument("--qlora", action="store_true", help="4-bit QLoRA (needs a CUDA GPU).")
    p.add_argument("--smoke", action="store_true", help="Tiny settings for a CPU loop smoke.")
    args = p.parse_args()

    config = TrainConfig(
        base_model=args.base,
        data_dir=args.data,
        output_dir=args.out,
        epochs=args.epochs,
        max_steps=args.max_steps,
        max_seq_len=args.max_seq_len,
        grad_accum=args.grad_accum,
        qlora=args.qlora,
    )
    if args.smoke:
        for key, value in SMOKE.items():
            setattr(config, key, value)

    out = train(config)
    print(f"saved adapter to {out}")


if __name__ == "__main__":
    main()
