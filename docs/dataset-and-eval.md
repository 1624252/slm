# Dataset & Evaluation

How the `islm` package generates the training dataset and evaluates base-vs-tuned models.
It implements PRD sections 10–14. The **validators are the backbone**: the same deterministic
checks filter the dataset, guide rewriting, grade the eval, and (later) guard inference.

Per `spec.md`, **the eval is built before any training run** — this whole package exists and
runs end-to-end (offline, via a mock teacher) before we fine-tune anything.

## Install

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1   |   macOS/Linux: source .venv/bin/activate
pip install -e .            # or: pip install -r requirements.txt
python -m spacy download en_core_web_sm   # recommended lemmatizer for real data
```

Everything except real teacher/judge calls runs with **no API key and no spaCy model**: heavy
deps are imported lazily, and a `SimpleLemmatizer` + `MockLLM` cover the offline path.

For real generation, copy `.env.example` to `.env` and add your key (the client is
OpenAI-compatible, so `OPENAI_BASE_URL` can point at any provider or a local server). **Never
commit `.env`.**

## Languages

Shipped: **English (`en`), Chinese (`zh`), Japanese (`ja`)**. The pipeline is language-agnostic —
any other code falls back to `wordfreq` frequency bands + a generic Unicode tokenizer.

| Language | Analyzer | Graded scheme | Known / to-learn tiers |
| --- | --- | --- | --- |
| en | spaCy (`en_core_web_sm`) or rule-based fallback | CEFR | A1–A2 / B1–C1 |
| zh | jieba | HSK | HSK1–3 / HSK4–6 |
| ja | fugashi + UniDic | JLPT | N5–N4 / N3–N1 |
| other | generic Unicode tokenizer | frequency | top-N / next band |

Vocabulary lives at `data/vocab/<lang>/{baseline,advanced}.csv` (`word,tier,source`). Small
curated samples ship for offline use and tests; regenerate or expand from frequency with:

```bash
python -m islm.vocab.build_lists --language ja --from-frequency --overwrite
```

`jieba`, `fugashi`, and `unidic-lite` install via `pip install -e .`. Without them, ZH/JA fall
back to a crude per-character tokenizer (fine for the smoke test, not for real data). Provenance
and licenses of the graded lists are recorded in `data/vocab/SOURCES.md`.

## Package layout

```
src/islm/
├── config.py            # Thresholds (from the PRD) + LLM settings from env
├── vocab/               # tokenize, lemmatize (spaCy or rule-based), word lists
├── validators/          # coverage, one-new-word, recurrence -> ValidationReport  (the backbone)
├── llm/                 # OpenAI-compatible client, offline MockLLM, prompt templates
├── datagen/             # scenario sampler, generate + rewrite loop, dataset pipeline
└── eval/                # LLM judge, cloze inferability, base-vs-tuned harness + report
```

## The Behavior Spec, in code

A story passes the deterministic gate (`ValidationReport.hard_pass`) when:

| Check | Module | Threshold (default) |
| --- | --- | --- |
| OOV rate / coverage | `validators/coverage.py` | OOV ≤ 1%, coverage ≥ 98% |
| ≤1 new word per sentence | `validators/one_new_word.py` | max 1 (repeats don't count as new) |
| Target recurrence | `validators/recurrence.py` | each target ≥ 3× |

Thresholds live in `islm.config.Thresholds`. The LLM judge (rubric) and cloze inferability are
**secondary** signals in `eval/`, because human↔judge correlation is only moderate (PRD 14.6).

## Generating the dataset

Pipeline (`datagen/pipeline.py`), one scenario → one example:

1. **Sample** a scenario: level → known list, 1–2 target words (not in the known list), a theme
   (`datagen/scenarios.py`).
2. **Generate** a story with the teacher (`generation_prompt` → `llm` client).
3. **Validate** with the deterministic backbone.
4. **Rewrite loop** (≤5 passes) feeding failures back to the model until `hard_pass` — no hard
   constrained decoding (SRS-Stories found it worst).
5. **Judge** (optional) with a different model family; keep only examples that pass the hard gate
   *and* the judge threshold.
6. **Write** `train/val/test.jsonl` (scenario-level split) + `stats.json`.

Offline smoke (mock teacher, no key); add `--language zh` or `--language ja` for other languages:

```bash
python -m islm.datagen.pipeline --n 20 --language en --mock --out data/generated/en
python -m islm.datagen.pipeline --n 20 --language zh --mock --out data/generated/zh
```

Real run (teacher + judge from `.env`):

```bash
python -m islm.datagen.pipeline --n 500 --language zh --model gpt-5 --judge-model claude-sonnet-5
```

### Record schema (JSONL)

```json
{
  "id": "en-A2-0007",
  "language": "en", "level": "A2", "theme": "a cat detective",
  "target_words": ["clue"],
  "messages": [
    {"role": "system", "content": "<rules>"},
    {"role": "user", "content": "Level/Theme/TARGET_WORDS/KNOWN_WORDS ..."},
    {"role": "assistant", "content": "<story>"}
  ],
  "metadata": {
    "sentences": 12, "oov_rate": 0.0, "coverage": 1.0,
    "max_new_words_per_sentence": 1, "target_recurrence": {"clue": 4},
    "rewrite_passes": 1, "judge_scores": {"spec_adherence": 2, "...": 2},
    "hard_pass": true, "kept": true, "split": "train"
  }
}
```

The training input (`system` + `user`) equals the inference input, so the model learns to write
within a provided allowed list (as in SRS-Stories). Generated data goes to `data/generated/`
(git-ignored; publish the final dataset to the HF Hub).

## Evaluating (base vs tuned)

`eval/harness.py` runs any *scenario → story* function over the held-out set and scores each
story with the validators + optional judge + optional cloze. `eval/report.py` renders the
base-vs-tuned table; the win condition (PRD 15) is **tuned beats base on hard-check pass rate and
judge spec-adherence**.

Offline smoke (mock as both models — proves the harness end-to-end); pick a language:

```bash
python -m islm.eval.run --mock --language zh
```

Real comparison (prompted base vs your fine-tuned model):

```bash
python -m islm.eval.run \
  --base-model qwen3-4b-instruct \
  --tuned-model my-tuned-model \
  --judge-model gpt-5
```

Held-out scenarios are read from `evals/scenarios/heldout_<lang>.jsonl` (committed for
reproducibility; auto-created on first run if missing). Results go to `evals/results/`
(git-ignored).

## Testing & linting

```bash
python -m pytest        # unit tests for the backbone + an offline end-to-end smoke test
ruff check src tests    # lint
ruff format src tests   # format
```

Tests use `SimpleLemmatizer` + `MockLLM`, so they are deterministic and need no network, key, or
model download.

## Notes & caveats

- **Lemmatizer:** `SimpleLemmatizer` is a rule-based fallback (imperfect on silent-e verbs);
  install spaCy for real data. Coverage checks both the lemma and the raw surface form, so a
  base-form word is never falsely flagged.
- **Known vocabulary** can come from the bundled `data/vocab/sample_en.csv`, a CEFR-J CSV, or
  frequency bands (`Vocabulary.from_frequency`, via `wordfreq`).
- **Mock output is intentionally minimal** ("The cat is big.") — it exists to exercise the
  pipeline, not to be good writing. A real teacher produces real stories; the validators are
  identical either way.
- **Chinese/Japanese** need a segmenter (jieba / fugashi); coverage matches on lemma **or**
  surface, which absorbs most segmentation and dictionary-form mismatches.
- **Any other language** works via `wordfreq` bands + a generic tokenizer — lower quality, but
  the same validators and eval apply unchanged.
