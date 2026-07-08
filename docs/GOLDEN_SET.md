# Golden set (Layer 1)

The golden set is the project's **first line of defense**: a small, held-out set of trusted
input→output pairs that must *all* pass on every commit. If a golden case fails, something is
fundamentally broken. This is Layer 1 of the 6-layer eval model (see `docs/EVALUATION.md`).

## What it is

Each item pairs:

- **Input** — a scenario: `language`, `level`, `theme`/tone, `TARGET_WORDS` (the new words to
  teach), `KNOWN_WORDS` (the allowed vocabulary). Rendered into the same system+user prompt the
  model sees in training and eval.
- **Reference output** — a hand-authored story that satisfies the full Behavior Spec.
- **Metadata** — `tone`, `keywords`, `target_tier` (CEFR/HSK/JLPT/exam/core), coverage tags
  (`category`, `subcategory`, `difficulty`), and the deterministic validator metrics
  (`oov_rate`, `coverage`, `target_recurrence`, `hard_pass`).

## Format & count

- **File:** `evals/golden/golden.jsonl` — one JSON object per line, in the training record schema
  (`id`, `language`, `level`, `theme`, `target_words`, `messages` [system/user/assistant],
  `metadata`). A companion `evals/golden/golden_report.json` holds the counts.
- **Count:** **55 items** — **39 English, 8 Chinese, 8 Japanese**. (Within the 50–200 range; the
  PDF's 10–20 is a floor, we hold more to span three languages and difficulty bands.) English
  spans all three difficulty levels (1/2/3-target); zh/ja are single-target so far — a visible gap
  in the coverage matrix (`evals/golden/coverage.md`), which is the point of Layer 2.
- **Provenance:** hand-authored (`source: golden-authored`), **distinct from the training seed**
  (`datagen/seed.py`) so scoring a model on the golden set never leaks training data. Built and
  validated by `src/islm/eval/golden.py`; every story is checked on build and only spec-passing
  stories are written.

## The four check-types (PDF) → our checks

The PDF lists four golden-set check types. Ours are the deterministic validators
(`src/islm/validators/`, no LLM) — binary, zero API cost, same result every run:

| PDF check type | What it catches | Our deterministic check |
| --- | --- | --- |
| Content validation | Response missing key facts | **coverage / OOV**: every word ∈ known ∪ target (`coverage.py`) |
| Tool / spec selection | Wrong behavior | **one-new-word**: ≤1 new word per sentence (`one_new_word.py`) |
| Source citation | Cited the wrong thing | **recurrence**: each target repeated ≥3× (`recurrence.py`) |
| Negative validation | Hallucinated / gave up | **hard_pass**: all of the above at once, else fail |

## Build & run

```bash
# Build (validates every story; writes golden.jsonl + report):
python -m islm.eval.golden --out evals/golden

# Just the breakdown, no write:
python -m islm.eval.golden --stats

# Run the regression gate (all must pass) — this is the every-commit check:
python -m pytest tests/test_golden.py -v
```

`tests/test_golden.py` re-runs the validators on every stored story (not just trusting the
metadata), checks the size range, the required metadata, that all three languages and the exam
tier are represented, and that no gold story leaks from the training seed.

## Rules (PDF Layer-1 discipline)

- **Run on every commit** — it's a regression test (wired via `tests/test_golden.py`).
- **Add from real failures** — when a model failure is found, add a golden case for it; don't bloat.
- **Never** change a reference output just to make a test pass — fix the model or the data instead.
