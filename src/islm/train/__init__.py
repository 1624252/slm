"""Supervised fine-tuning (LoRA / QLoRA) on the generated i+1 story dataset."""

# Lazy re-export: eagerly importing `.sft` here makes `python -m islm.train.sft` double-import the
# module (package init loads it, then runpy runs it as __main__) and emit a RuntimeWarning. Defer
# via __getattr__ so `from islm.train import train` still works, without the eager import.
__all__ = ["SMOKE", "TrainConfig", "load_texts", "render_plain", "train"]


def __getattr__(name: str):
    if name in __all__:
        from . import sft

        return getattr(sft, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
