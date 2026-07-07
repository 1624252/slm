# Data card — i+1 Story dataset

What the dataset is, how it's generated, how it's filtered (the two-pass plan), and how to
reproduce/scale it. Companion to `docs/dataset-and-eval.md` (mechanics) and `data/vocab/SOURCES.md`
(vocabulary licenses).

## What it is

Supervised fine-tuning examples for comprehensible-input (i+1) story generation. One example =
one story that, for a given known-vocabulary list `K` and target words `T`, stays ≥98% inside
`K`, adds ≤1 new word per sentence, makes each new word inferable, and repeats each target ≥3×.

- **Languages:** English, Chinese (Mandarin), Japanese; any other language works via frequency
  bands + a generic tokenizer.
- **Format:** JSONL, one record per line (schema below); chat-formatted for SFT.
- **Unit / label:** the assistant story is the target; the system+user messages are the input.

## How it's generated (two passes)

### Pass 1 — generation (`islm.datagen.pipeline`)

Per scenario `(language, K, T, theme)`:

1. **Sample** a scenario (`datagen/scenarios.py`): `K` = the language's baseline tier, `T` =
   1–2 words from the advanced tier (not in `K`), a narrow-reading theme.
2. **Generate** a story with a frontier **teacher** model (`llm/` client; prompt in
   `llm/prompts.py`).
3. **Validate** with the deterministic backbone (coverage/OOV, ≤1-new-word, recurrence).
4. **Rewrite loop** (≤5 passes) feeding failures back to the teacher until `hard_pass` — no hard
   constrained decoding (SRS-Stories found it worst).
5. **Per-item judge** (optional, different model family): keep only stories that pass the hard
   gate *and* the judge threshold.

Output: `train/val/test.jsonl` (scenario-level split) + `stats.json`.

### Pass 2 — curation (`islm.datagen.curate`)

The plan is to generate **a lot** (thousands), then review the whole corpus and drop the bad
ones. This pass runs corpus-wide checks that per-item generation can't:

| Reason (good = none) | What it catches |
| --- | --- |
| `duplicate` / `near_duplicate` | exact and structural repeats (teachers repeat at scale) |
| `repetitive_sentences` | low unique-sentence ratio (filler, not a story) |
| `low_lexical_variety` | low type-token ratio |
| `too_short` | fewer than the minimum sentences |
| `failed_revalidation` | re-running the hard validators as defense-in-depth |
| `judge_spec_adherence` / `judge_low_quality` | stricter optional LLM re-judge |

Output: filtered `train/val/test.jsonl` + `rejects.jsonl` (annotated with reasons) +
`curation_report.{md,json}`. Thresholds live in `CurationConfig`.

> Note: the deterministic curation reliably removes duplicates, degenerate repetition, and
> coverage failures. Separating "passes the rules but is dull/incoherent" from genuinely good
> prose needs the **LLM re-judge** — i.e. a teacher/judge model. That is why the large-scale
> good dataset depends on model access (below).

## Record schema (JSONL)

```json
{
  "id": "en-0007", "language": "en", "level": "A1-A2", "theme": "a cat detective",
  "target_words": ["clue"],
  "messages": [
    {"role": "system", "content": "<rules>"},
    {"role": "user", "content": "Language/Level/Theme/TARGET_WORDS/KNOWN_WORDS ..."},
    {"role": "assistant", "content": "<story>"}
  ],
  "metadata": {
    "sentences": 8, "oov_rate": 0.0, "coverage": 1.0,
    "max_new_words_per_sentence": 1, "target_recurrence": {"clue": 4},
    "rewrite_passes": 1, "judge_scores": {"spec_adherence": 2, "...": 2},
    "hard_pass": true, "kept": true, "split": "train"
  }
}
```

## Vocabulary provenance

Known/target tiers come from graded lists (CEFR-J/Octanove, HSK 3.0, JLPT) or frequency bands
(`wordfreq`). Exact datasets and licenses: `data/vocab/SOURCES.md`. Downloaded full lists are
git-ignored; small curated CC0 samples ship in the repo.

## Current status

| Artifact | Status |
| --- | --- |
| Vocabulary (en/zh/ja) | Real graded lists fetched (en ~2.3k/6.4k, zh ~2.2k/8.7k, ja ~1.3k/6.7k baseline/advanced) |
| Curated CC0 samples | Committed (offline + tests) |
| Human-authored English seed | 6 spec-passing stories (`islm.datagen.seed`), curation keeps 6/7 (drops 1 planted duplicate) |
| Large teacher-generated corpus | **Pending model access** (see below) |

The authored seed is genuine, spec-passing data that also exercises the two-pass curation on real
prose. The thousands-scale corpus requires a teacher model.

## Reproduce / scale

```bash
# 1. Real vocabulary
python -m islm.vocab.download --language all

# 2. Human-authored English seed (no model needed)
python -m islm.datagen.seed --out data/generated/en_seed

# 3. Large generation with a teacher (needs .env with an OpenAI-compatible key)
python -m islm.datagen.pipeline --n 4000 --language en \
    --model <teacher> --judge-model <judge> --out data/generated/en

# 4. Second-pass curation
python -m islm.datagen.curate --in data/generated/en --out data/curated/en --judge-model <judge>
```

## Limitations & ethics

- **Model access required for scale.** Without a teacher, only the mock (filler) and the
  hand-authored seed exist; mock filler is not good data and is not shipped as such.
- **Vocabulary licenses:** CC-BY-SA-4.0 items require attribution + ShareAlike; CEFR-J requires
  citation; JLPT levels are an unofficial community compilation. See `SOURCES.md`.
- **Segmentation:** Chinese/Japanese depend on jieba/fugashi; coverage matches lemma **or**
  surface to absorb most segmentation/dictionary-form mismatches.
- **Judge reliability** is only moderate (PRD 14.6): hard validators are primary, the judge is a
  secondary quality signal, and a human should spot-check the final set.
