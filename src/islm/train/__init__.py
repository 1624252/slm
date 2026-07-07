"""Supervised fine-tuning (LoRA / QLoRA) on the generated i+1 story dataset."""

from .sft import SMOKE, TrainConfig, load_texts, render_plain, train

__all__ = ["SMOKE", "TrainConfig", "load_texts", "render_plain", "train"]
