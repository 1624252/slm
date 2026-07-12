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
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "dataset_v2"
DATA_V1 = ROOT / "data" / "dataset_v1"


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


def _normalize_record(rec: dict) -> str:
    """Make one record HF/Arrow-safe: a stable metadata schema across every row. The v1 corpus
    stores `target_recurrence` as a dict keyed by per-record target words, which gives each row a
    different struct type and breaks parquet conversion (the same failure v2 fixed by stringifying
    it). Serialize any dict-valued metadata field to a JSON string so the column type stays fixed.
    """
    meta = rec.get("metadata", {})
    for key in ("target_recurrence", "judge_scores"):
        if isinstance(meta.get(key), dict):
            meta[key] = json.dumps(meta[key], ensure_ascii=False)
    return json.dumps(rec, ensure_ascii=False)


def _stage_split(src_dir: Path, split: str, dest: Path, normalize: bool = False) -> None:
    """Read a split (plain .jsonl or .jsonl.gz) from src_dir and write it into dest, optionally
    normalizing each record's metadata for a stable Arrow schema."""
    plain, gz = src_dir / f"{split}.jsonl", src_dir / f"{split}.jsonl.gz"
    if plain.exists():
        text = plain.read_text(encoding="utf-8")
    elif gz.exists():
        text = gzip.decompress(gz.read_bytes()).decode("utf-8")
    else:
        raise SystemExit(f"missing split: {split} in {src_dir}")
    if normalize:
        lines = [_normalize_record(json.loads(ln)) for ln in text.splitlines() if ln.strip()]
        text = "\n".join(lines) + "\n"
    dest.write_text(text, encoding="utf-8")


def publish_dataset(repo: str, token: str) -> None:
    api = _api(token)
    api.create_repo(repo, repo_type="dataset", exist_ok=True)
    # Stage two configs: `default` = the curated v2 splits at the repo root; `v1` = the pre-v2
    # (templated) corpus under v1/, kept as its own browsable config so the viewer stays working
    # and the curated set stays front-and-center. v1 records normalized (dict metadata -> string).
    stage = ROOT / ".hf_stage_dataset"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir()
    (stage / "v1").mkdir()
    for split in ("train", "val", "test"):
        _stage_split(DATA, split, stage / f"{split}.jsonl")
        _stage_split(DATA_V1, split, stage / "v1" / f"{split}.jsonl", normalize=True)

    stats_raw = (DATA / "stats.json").read_text(encoding="utf-8")
    (stage / "stats.json").write_text(stats_raw, encoding="utf-8")
    (stage / "v1" / "stats.json").write_text(
        (DATA_V1 / "stats.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    stats = json.loads(stats_raw)
    total = stats.get("total_elements", "?")
    by_split = stats.get("by_split", {})
    by_lang = stats.get("by_language", {})
    v1_stats = json.loads((DATA_V1 / "stats.json").read_text(encoding="utf-8"))
    v1_total = v1_stats.get("total_elements", "?")

    # README front-matter: license/language/tags + a configs block. Two configs: `default` (v2,
    # curated) and `v1` (pre-v2, templated) so the Hub viewer shows both with their own splits.
    fm = [
        "---",
        "license: mit",
        "language: [en, zh, ja]",
        "task_categories: [text-generation]",
        "tags: [comprehensible-input, language-learning, i-plus-1, story-generation]",
        "pretty_name: i+1 Story Dataset (en/zh/ja)",
        "configs:",
        "  - config_name: default",
        "    data_files:",
        "      - {split: train, path: train.jsonl}",
        "      - {split: validation, path: val.jsonl}",
        "      - {split: test, path: test.jsonl}",
        "  - config_name: v1",
        "    data_files:",
        "      - {split: train, path: v1/train.jsonl}",
        "      - {split: validation, path: v1/val.jsonl}",
        "      - {split: test, path: v1/test.jsonl}",
        "---",
        "",
    ]
    # A human-readable "at a glance" block built from stats.json, above the full data card.
    glance = [
        "## Dataset at a glance",
        "",
        "**`default` config — curated v2 (the recommended set):**",
        "",
        f"- **Total examples:** {total}",
        "- **Splits:** " + ", ".join(f"{k} {v}" for k, v in by_split.items()),
        "- **By language:** " + ", ".join(f"{k} {v}" for k, v in by_lang.items()),
        "",
        "**`v1` config — pre-v2 (templated, superseded):**",
        "",
        f"- **Total examples:** {v1_total}",
        "- **Splits:** " + ", ".join(f"{k} {v}" for k, v in v1_stats.get("by_split", {}).items()),
        "- **By language:** "
        + ", ".join(f"{k} {v}" for k, v in v1_stats.get("by_language", {}).items()),
        "",
        "> v1 is the programmatic/templated corpus that the project's thesis argues against; v2 is"
        " the curated fix. Included for before/after comparison, not for training the final model.",
        "",
    ]
    body = (ROOT / "docs" / "DATA_CARD.md").read_text(encoding="utf-8")
    readme = "\n".join(fm) + "\n".join(glance) + "\n" + body
    (stage / "README.md").write_text(readme, encoding="utf-8")
    api.upload_folder(folder_path=str(stage), repo_id=repo, repo_type="dataset")
    shutil.rmtree(stage)
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
    # Upload ONLY the final adapter — not the checkpoint-*/ scratch (optimizer/RNG state, huge and
    # useless for inference). delete_patterns also strips any stale checkpoints already on the repo.
    api.upload_folder(
        folder_path=str(adapter),
        repo_id=repo,
        repo_type="model",
        allow_patterns=[
            "adapter_config.json",
            "adapter_model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "chat_template.jinja",
            "train_summary.json",
            "README.md",
        ],
        delete_patterns=["checkpoint-*/*", "checkpoint-*", "*.pt", "*.pth", "optimizer*", "*.bin"],
    )
    print(f"model published (adapter only) -> https://huggingface.co/{repo}")


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
