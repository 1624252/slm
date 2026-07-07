"""Training data-format tests; the heavy train() is smoke-run separately (no torch/trl needed)."""

import json

from islm.train.sft import SMOKE, TrainConfig, load_texts, read_records, render_plain


def test_render_plain():
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ]
    assert render_plain(messages) == "system: rules\nuser: u\nassistant: a"


def test_read_and_load_texts(tmp_path):
    record = {
        "messages": [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U"},
            {"role": "assistant", "content": "A"},
        ]
    }
    (tmp_path / "train.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    assert len(read_records(tmp_path)) == 1
    # No tokenizer -> plain rendering, so the loop is testable offline.
    assert load_texts(tmp_path) == ["system: S\nuser: U\nassistant: A"]


def test_smoke_config_applies():
    config = TrainConfig(base_model="m", data_dir=".", output_dir=".")
    for key, value in SMOKE.items():
        setattr(config, key, value)
    assert config.max_steps == 3
    assert config.max_seq_len == 256
