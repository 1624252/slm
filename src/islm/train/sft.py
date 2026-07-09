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
    # Optimizer/schedule — matches the standard QLoRA recipe (cosine + warmup + clip + decay).
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.001
    max_grad_norm: float = 0.3
    gradient_checkpointing: bool = True
    merge: bool = False  # after training, merge the adapter into fp16 base (for upload/deploy)
    resume_adapter: str | None = None  # continue an existing adapter (local path or HF hub id)
    push_to_hub: str | None = None  # after training, push adapter+tokenizer to this HF Hub repo id


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

    if config.resume_adapter:
        # Continue an existing adapter (keep training the best model) instead of a fresh one.
        from peft import PeftModel

        if config.qlora and on_gpu:
            from peft import prepare_model_for_kbit_training

            model = prepare_model_for_kbit_training(model)
        model = PeftModel.from_pretrained(model, config.resume_adapter, is_trainable=True)
        peft_config = None  # model is already a PEFT model; don't wrap a new adapter
        print(f"resuming training from adapter: {config.resume_adapter}")
    else:
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
    # paged_adamw_32bit needs bitsandbytes (GPU); fall back to plain adamw on CPU.
    optim = "paged_adamw_32bit" if (config.qlora and on_gpu) else "adamw_torch"
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
        # Training-stability settings from the standard QLoRA recipe.
        optim=optim,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        gradient_checkpointing=config.gradient_checkpointing and on_gpu,
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
        "lr_scheduler_type": config.lr_scheduler_type,
        "warmup_ratio": config.warmup_ratio,
        "weight_decay": config.weight_decay,
        "max_grad_norm": config.max_grad_norm,
        "optim": optim,
        "final_train_loss": round(train_loss, 4) if train_loss is not None else None,
    }
    with open(config.output_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"final train loss: {summary['final_train_loss']}")

    if config.push_to_hub:
        _push_to_hub(config)
    if config.merge:
        _merge_adapter(config)
    return config.output_dir


def _push_to_hub(config: TrainConfig) -> None:
    """Push the saved adapter + tokenizer to an HF Hub repo (needs HF_TOKEN with write scope).

    Uploads the adapter files but skips `checkpoint-*/` (optimizer state) to keep the repo small.
    The pushed adapter can later be continued with `--resume-adapter <repo_id>`.
    """
    import os

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("push-to-hub skipped: set HF_TOKEN (write scope) to enable.")
        return
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(config.push_to_hub, private=True, exist_ok=True)
    api.upload_folder(
        repo_id=config.push_to_hub,
        folder_path=str(config.output_dir),
        ignore_patterns=["checkpoint-*", "checkpoint-*/*"],  # skip optimizer state
    )
    print(f"pushed adapter -> https://huggingface.co/{config.push_to_hub}")


def _merge_adapter(config: TrainConfig) -> Path:
    """Merge the LoRA adapter into the base weights → a standalone fp16 model (for upload/deploy).

    Mirrors the reference notebook's final step: reload the base in fp16, apply the adapter,
    `merge_and_unload()`, and save to `<output_dir>-merged`. Skipped by default (adapters are
    smaller); use --merge on the GPU when you want a self-contained model.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    merged_dir = Path(str(config.output_dir) + "-merged")
    base = AutoModelForCausalLM.from_pretrained(
        config.base_model, low_cpu_mem_usage=True, return_dict=True, torch_dtype=torch.float16
    )
    model = PeftModel.from_pretrained(base, str(config.output_dir)).merge_and_unload()
    model.save_pretrained(str(merged_dir))
    AutoTokenizer.from_pretrained(config.base_model).save_pretrained(str(merged_dir))
    print(f"merged model -> {merged_dir}")
    return merged_dir


def main() -> None:
    p = argparse.ArgumentParser(description="SFT (QLoRA/LoRA) for the i+1 story model.")
    p.add_argument("--data", type=Path, required=True, help="Dataset dir with train.jsonl.")
    p.add_argument("--base", required=True, help="Base model (HF path or hub id).")
    p.add_argument("--out", type=Path, default=Path("outputs/lora"))
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--max-seq-len", type=int, default=1024, help="Context window (left-truncated).")
    p.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps.")
    p.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    p.add_argument("--lora-r", type=int, default=16, help="LoRA rank.")
    p.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha.")
    p.add_argument("--qlora", action="store_true", help="4-bit QLoRA (needs a CUDA GPU).")
    p.add_argument("--merge", action="store_true", help="Merge adapter into fp16 base after train.")
    p.add_argument(
        "--resume-adapter", default=None,
        help="Continue an existing adapter (local path or HF hub id) instead of a fresh one.",
    )
    p.add_argument(
        "--push-to-hub", default=None,
        help="Push the adapter to this HF Hub repo id after training (needs HF_TOKEN write scope).",
    )
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
        learning_rate=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        qlora=args.qlora,
        merge=args.merge,
        resume_adapter=args.resume_adapter,
        push_to_hub=args.push_to_hub,
    )
    if args.smoke:
        for key, value in SMOKE.items():
            setattr(config, key, value)

    out = train(config)
    print(f"saved adapter to {out}")


if __name__ == "__main__":
    main()
