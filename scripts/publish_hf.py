"""Publish the dataset and/or the fine-tuned adapter to the Hugging Face Hub.

Run this once your HF account is verified (the earlier create_repo 500s were an account-level block,
almost certainly an unverified email — confirm at https://huggingface.co/settings/account first).

    # dataset (defaults to repo i0445/islm-stories; created on first push):
    HF_TOKEN=... python scripts/publish_hf.py dataset

    # model/adapter (defaults to repo i0445/islm; point --adapter at the LoRA dir):
    HF_TOKEN=... python scripts/publish_hf.py model --adapter path/to/qwen3_4b_v2_multi

    # override either default with --repo <namespace>/<name>.

`HF_TOKEN` is read from the environment or `.env` (never hard-code it). Both subcommands create the
repo if missing and are safe to re-run (uploads overwrite).
"""

from __future__ import annotations

import argparse
import gzip
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "dataset_v2"


def _token() -> str:
    tok = os.environ.get("HF_TOKEN")
    if not tok:  # fall back to .env
        env = ROOT / ".env"
        if env.exists():
            for line in env.read_text(encoding="utf-8").splitlines():
                if line.startswith("HF_TOKEN="):
                    tok = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not tok:
        sys.exit("HF_TOKEN not set (env or .env). Get one at https://huggingface.co/settings/tokens")
    return tok


def _api(token: str):
    from huggingface_hub import HfApi

    return HfApi(token=token)


def publish_dataset(repo: str, token: str) -> None:
    api = _api(token)
    api.create_repo(repo, repo_type="dataset", exist_ok=True)
    # Decompress the shipped gz splits into a temp staging dir, upload with stats + data card.
    stage = ROOT / ".hf_stage_dataset"
    stage.mkdir(exist_ok=True)
    for split in ("train", "val", "test"):
        gz = DATA / f"{split}.jsonl.gz"
        (stage / f"{split}.jsonl").write_text(
            gzip.decompress(gz.read_bytes()).decode("utf-8"), encoding="utf-8"
        )
    stats = (DATA / "stats.json").read_text(encoding="utf-8")
    (stage / "stats.json").write_text(stats, encoding="utf-8")
    # README = the data card, so the Hub dataset page is populated.
    (stage / "README.md").write_text(
        (ROOT / "docs" / "DATA_CARD.md").read_text(encoding="utf-8"), encoding="utf-8"
    )
    api.upload_folder(folder_path=str(stage), repo_id=repo, repo_type="dataset")
    for f in stage.iterdir():
        f.unlink()
    stage.rmdir()
    print(f"dataset published -> https://huggingface.co/datasets/{repo}")


def publish_model(repo: str, adapter: Path, base: str, token: str) -> None:
    if not adapter.is_dir():
        sys.exit(f"adapter dir not found: {adapter} (download it from Drive first)")
    api = _api(token)
    api.create_repo(repo, repo_type="model", exist_ok=True)
    # Minimal model card so the Hub page explains what this is and how to load it.
    card = f"""---
base_model: {base}
library_name: peft
tags: [qlora, lora, comprehensible-input, language-learning, i-plus-1]
---

# i+1 Story SLM — QLoRA adapter

LoRA adapter over `{base}` that writes comprehensible-input (i+1) language-learning stories in
English, Chinese, and Japanese: every story stays inside a known-vocabulary set, adds at most one
new word per sentence, recurs each target word >=3x, keeps new words inferable, and reads as a real
story. See the dataset and eval harness for the behavior spec and base-vs-tuned numbers.

## Load

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base = "{base}"
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, device_map="auto")
model = PeftModel.from_pretrained(model, "{repo}")
```
"""
    (adapter / "README.md").write_text(card, encoding="utf-8")
    api.upload_folder(folder_path=str(adapter), repo_id=repo, repo_type="model")
    print(f"model published -> https://huggingface.co/{repo}")


def main() -> None:
    p = argparse.ArgumentParser(description="Publish dataset/model to the HF Hub.")
    sub = p.add_subparsers(dest="what", required=True)
    d = sub.add_parser("dataset")
    d.add_argument("--repo", default="i0445/islm-stories", help="HF dataset repo")
    m = sub.add_parser("model")
    m.add_argument("--repo", default="i0445/islm", help="HF model repo (default: i0445/islm)")
    m.add_argument("--adapter", type=Path, required=True, help="local LoRA adapter dir")
    m.add_argument("--base", default="Qwen/Qwen3-4B-Instruct-2507")
    args = p.parse_args()

    token = _token()
    if args.what == "dataset":
        publish_dataset(args.repo, token)
    else:
        publish_model(args.repo, args.adapter, args.base, token)


if __name__ == "__main__":
    main()
