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

**Two versions, two HF configs.** The published dataset ships both:

- **`default`** — the curated **v2** set (1,610 stories). The recommended set; what the final model
  trains on.
- **`v1`** — the pre-v2 **templated** corpus (144,748 stories). Superseded, included for the
  before/after comparison the project's thesis rests on. Load with
  `load_dataset("i0445/islm-stories", "v1")`.

## How it's generated (two passes)

### Pass 1 — generation (`islm.datagen.pipeline`)

Per scenario `(language, K, T, theme)`:

1. **Sample** a scenario (`datagen/scenarios.py`): `K` = the language's baseline tier, `T` =
   1–2 words from the advanced tier (not in `K`), a narrow-reading theme.
2. **Generate** a story with a frontier **teacher** model (`llm/` client; prompt in
   `llm/prompts.py`).
3. **Validate** with the deterministic backbone (coverage/OOV — out-of-vocabulary words not in
   `K ∪ T` — ≤1-new-word, recurrence).
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
| Human-authored seed (en/zh/ja) | 28 unique spec-passing stories via `islm.datagen.seed`; the hand-crafted quality core. |
| Teacher-generated corpus | ~80 curated records via `islm.datagen.pipeline` (frontier teacher); quality, small (API/time-bound). |
| **Dataset v1 (the large deliverable)** | **144,748 spec-passing records**, each teaching **2–3 target words** (en 114,753 / zh 15,000 / ja 14,995) via `islm.datagen.synth` (programmatic) + two-pass `curate`. See below. |

## Dataset v1 — the large corpus (`data/dataset_v1/`)

**144,748 elements** — train 115,801 / val 14,479 / test 14,468; by language en 114,753 / zh
15,000 / ja 14,995. **Each story teaches 2–3 target words** (avg 2.33; 96,710 two-target + 48,038
three-target → 337,534 total target mentions). **Every record is spec-passing** (validated: ≤2%
OOV, ≤1 new word/sentence, **each** target recurs ≥3×, and each target's first appearance is in its
own sentence so ≤1 new word/sentence holds). Same chat-record schema as the seed (system rules +
user scenario + assistant story; `metadata` carries validator scores, `source`, and `target_pos`
— a list, one POS per target).

**How it was built (the "make a lot, then filter" process):**
1. **First pass — programmatic generation** (`islm.datagen.synth`): compose stories from large
   pools of common words + story arcs, inserting one **POS-typed target** (noun/adj/verb) placed in
   a grammatically correct frame. Each story's `KNOWN_WORDS` is scoped to its own words (minus the
   target) so coverage passes by construction and the target is the only "new" word. Generated
   ~145k candidates.
2. **Second pass — curation** (`islm.datagen.curate`): dedup (exact + near-dup), reject degenerate
   repetition / too-short / low-lexical-variety, re-validate. Kept 144,748 (en 99.8% — multi-target
   composition makes near-duplicates vanishingly rare; only 1 near-dup + 246 low-variety dropped).
   **Zero train/test leakage** (dedup is global across splits).

**Honest characterization (read this).** This is **programmatically generated**, not
teacher-distilled — a deliberate choice to reach 100k+ in a day (real teacher distillation at this
scale is days + large API cost). Consequences:
- ✅ 100% spec-compliant i+1 data by construction; grammatical (POS-routed); no leakage.
- ⚠️ **Limited lexical/structural diversity**: it teaches **279 unique (language, target) pairs**
  (2–3 per story) across a handful of story arcs, so prose patterns repeat (varied by
  character/setting/targets, not by deep narrative variety). A model trained on it may learn the
  *template*, not open-ended storytelling. Multi-target composition does widen combinatorial
  variety substantially (each story now interleaves 2–3 target beats).
- ⚠️ CJK is **noun-target only** (safest for grammaticality); en covers noun/adj/verb.
- The teacher-distilled corpus (smaller, higher prose quality) and this synthetic corpus are
  **complementary**; the strongest training mix is teacher-core + synthetic-bulk.

## Reproduce / scale

```bash
# 1. Real vocabulary
python -m islm.vocab.download --language all

# 2. Human-authored seed (en/zh/ja; no model needed), then curate it
python -m islm.datagen.seed --out data/generated/seed
python -m islm.datagen.curate --in data/generated/seed --out data/curated/seed

# 3. Large generation with a teacher (needs .env with an OpenAI-compatible key)
python -m islm.datagen.pipeline --n 4000 --language en \
    --model <teacher> --judge-model <judge> --out data/generated/en
python -m islm.datagen.curate --in data/generated/en --out data/curated/en --judge-model <judge>

# 4. Dataset v1 — the 135k programmatic corpus (no API; fully reproducible via seeds)
python -m islm.datagen.synth --n 115000 --language en --seed 42 --out data/generated/synth_en
python -m islm.datagen.synth --n 15000  --language zh --seed 43 --out data/generated/synth_zh
python -m islm.datagen.synth --n 15000  --language ja --seed 44 --out data/generated/synth_ja
for L in en zh ja; do
  # --fast uses the rule-based lemmatizer (skip spaCy/jieba/fugashi); ~60x faster where spaCy
  # models are installed (e.g. Colab). Drop it to use the higher-fidelity analyzers.
  python -m islm.datagen.curate --in data/generated/synth_$L --out data/curated/synth_$L --fast
done
mkdir -p data/dataset_v1
for S in train val test; do
  cat data/curated/synth_{en,zh,ja}/$S.jsonl > data/dataset_v1/$S.jsonl
done
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
