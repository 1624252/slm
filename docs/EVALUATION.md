# Evaluation

How we measure whether the fine-tuned model actually beats the prompted base at the target
behavior. Per `spec.md`, the eval is **built before any training** — it is the make-or-break
piece. It implements the spec's minimum bar and Appendix A rubric (PRD 14–15).

## The six eval layers

The eval stack maps to the 6-layer eval maturity model ("Evals That Actually Work"). Each layer is
a concrete, shipped artifact in this repo:

| # | Layer | What it answers | Artifact | Run |
| --- | --- | --- | --- | --- |
| 1 | **Golden sets** | Does it work? (correctness) | `evals/golden/golden.jsonl` (55 held-out cases), `docs/GOLDEN_SET.md` | `pytest tests/test_golden.py` (every commit) |
| 2 | **Behavioral coverage** | Does it work for all types? | tagged golden cases → `evals/golden/coverage.md` | `python -m islm.eval.coverage_matrix` |
| 3 | **Error analysis** | Where does it break? | `docs/ERROR_ANALYSIS.md` taxonomy + `error_analysis()` tally | read traces weekly |
| 4 | **Replay harnesses** | Re-score without re-running | `src/islm/eval/replay.py` | `python -m islm.eval.replay --results ...` |
| 5 | **Rubrics** | How good, not just pass/fail | 8-dim anchored `_RUBRIC` (live judge: `claude-sonnet-4-6`) + `calibration.py` | judge runs by default in `eval.run`; `calibration --scores ...` |
| 6 | **Experiments** | Which change is better? | `evals/LEADERBOARD.md` + `RESULTS_LOG.md` + `--track` | `train-islm` skill (one var/run) |

**Binary first, scores second.** The deterministic validators (Layers 1–2) are the primary gate;
the LLM-judge rubric (Layer 5) is the quality layer and is trusted only after calibration ≥ 0.8.
Results also push to **LangSmith** for dataset/experiment tracking (`langsmith_sync.py`, augment
mode — the committed JSONL stays the source of truth).

## The Behavior Spec (what we grade against)

> For a given known-vocabulary list `K`, target-word set `T`, and theme, the model returns a short
> story in the target language such that: (1) every word is in `K ∪ T` (or a trivially inferable
> proper noun); (2) no sentence introduces more than one word from `T`; (3) each introduced target
> word's meaning is inferable from its sentence's context; (4) every target word in `T` recurs at
> least 3 times; and (5) the story is coherent and compelling, with any humor/surprise attached to
> a target word rather than an off-topic detail. It never states that it is teaching or labels the
> target words.

## What the eval tests

The spec's minimum bar is three things; we implement all three plus robustness.

| Spec requirement | Our implementation |
| --- | --- |
| A behavioral check for the specific failure the spec forbids | **Deterministic validators** (below) |
| An LLM-as-judge scoring outputs against the Behavior Spec | **Judge rubric** (Appendix A, below) |
| A base-vs-tuned comparison | **`islm.eval.run`** → results table with deltas |

### 1. Deterministic behavioral checks (`validators/`, the primary gate)

Run on every output; these are the failures the spec forbids. Targets from PRD 14.2.

**OOV** stands for **out-of-vocabulary**: a running word whose lemma (or surface form) is not in
the learner's known set `K` or the target set `T`. **OOV rate** = OOV words ÷ total words, and
**coverage** = 1 − OOV rate (the fraction of the story the learner can already understand). The
**ideal is 100% coverage** (0% OOV); the pass gate tolerates **OOV ≤ 2%** — at most 2 words in
100 outside the learner's vocabulary (equivalently coverage ≥ 98%, since coverage = 1 − OOV rate).

| Check | What it verifies | Target (tuned) |
| --- | --- | --- |
| Coverage / OOV | fraction of words outside `K ∪ T` (out-of-vocabulary) | ideal 100% coverage; gate OOV ≤ 2% (coverage ≥ 98%) |
| ≤1 new word / sentence | pacing of new vocabulary | ≤ 1 new word in **100% of sentences** |
| Recurrence | each target repeated for spaced repetition | each target ≥ 3× |
| Inferability (cloze) | new word guessable from context | ≥ 60% recovered |

Coverage matches on lemma **or** surface, and for CJK on character decomposition, so natural
Chinese/Japanese text isn't falsely flagged (see `docs/dataset-and-eval.md`).

### 2. LLM-as-judge rubric (spec Appendix A + project dims; each 0/1/2)

Every dimension has **falsifiable 0/1/2 anchors** in `llm/prompts.py` `_RUBRIC` (a concrete,
checkable condition per score — no vague "good quality"), so two judges land on the same number.

| Dimension | 0 | 1 | 2 |
| --- | --- | --- | --- |
| `spec_adherence` | ≥2 spec rules broken | one rule broken once | all words allowed, ≤1 new/sentence, targets recur ≥3× |
| `robustness` | abandons behavior under pressure | leaks 1–2 OOV | holds every rule despite a hard prompt |
| `task_quality` | incoherent / degenerate loop | readable but flat | coherent arc a learner enjoys |
| `consistency` | varies sentence to sentence | one clear drop-off | same behavior first to last |
| `inferability` | targets not guessable | one needs outside knowledge | every new word guessable from context |
| `seductive_detail_control` | announces / labels the lesson | interest sits off-target | humor lands on the target word |
| `coherence` (new) | disjointed / self-contradicting | one non-sequitur | logical flow, consistent setting |
| `interestingness` (new) | dull / mechanical filler | pleasant but predictable | a real hook, surprise, or vivid image |

The first four are the **spec dimensions** — the win condition is judged on them. `coherence`
(does the story hold together?) and `interestingness` (is it worth reading?) were added as
separate signals so "compelling input" is measured, not assumed. The judge is a **secondary**
signal — human↔judge correlation is only moderate (SRS-Stories r≈0.46–0.56), so the deterministic
checks stay primary and the judge is trusted only after calibration ≥ 0.8 (`calibration.py`).

**The judge model.** Set by `JUDGE_MODEL` in `.env` (default `claude-group/claude-sonnet-4-6` via
the OpenAI-compatible endpoint in `OPENAI_BASE_URL`). It runs **by default** whenever a key is
present; `--no-judge` forces the deterministic-only path, and `--judge-model` overrides the model.
Deliberately a different family from the teacher, to reduce self-preference bias (PRD 11).

### 3. Robustness / adversarial (spec Appendix A "Robustness", PRD 14.5)

`eval/adversarial.py` builds scenarios that stress the behavior: a **small** known vocabulary,
**multiple, harder** targets, and **jargon-tempting** themes. Robustness = the hard-check pass
rate on this set. A brittle model leaks OOV words here even if it looks fine on the clean set.

## Reading the metrics (what each value means)

All rates are fractions in **[0, 1]**; judge scores are in **[0, 2]**. In the base-vs-tuned
tables, **Delta = tuned − base** and **Better** names the model that wins that row.

| Metric (as shown in results) | Range | What the value means | Better |
| --- | --- | --- | --- |
| Hard-check pass rate | 0–1 | fraction of stories passing **all** deterministic checks (coverage + ≤1-new-word + recurrence). 1.0 = every story is spec-compliant. | higher |
| OOV rate | 0–1 | fraction of words out-of-vocabulary (not in `K ∪ T`). 0.0 = ideal; pass gate ≤ 0.02. | lower |
| Coverage (= 1 − OOV) | 0–1 | fraction of words the learner already knows. 1.0 = ideal. | higher |
| ≤1 new word/sentence | 0–1 | fraction of stories in which **every** sentence adds ≤ 1 new word. 1.0 = all comply. | higher |
| Recurrence satisfied | 0–1 | fraction of stories in which **every** target word recurs ≥ 3×. | higher |
| Inferability (cloze) | 0–1 | fraction of target words a model recovers when the word is blanked (context-guessability). | higher |
| Judge: `<dimension>` | 0–2 | mean rubric score across judged stories (0 = fails, 1 = partial, 2 = fully). | higher |
| Judge: overall | 0–2 | mean across all judge dimensions. | higher |
| Adversarial hard-check pass | 0–1 | hard-check pass rate on the stress set (this **is** the robustness number). | higher |

Worked example (the committed English base): hard-check pass **0.000** (no story compliant),
OOV **0.413** (41% of words unknown to the learner), ≤1-new-word **0.000** (no story kept the
pacing), recurrence **0.125** (only ~1 story in 8 repeated its targets enough) — the base fails
the behavior on every axis, which is the baseline fine-tuning must beat.

## Required outputs & win condition

`islm.eval.run` writes `evals/results/results_<lang>.{md,json}` with:

- Mean of each metric and each judge dimension, **base vs tuned, with deltas**, on the same
  held-out scenarios.
- The **robustness** table (adversarial hard-check pass, base vs tuned).
- A **win-condition verdict** — the spec's bar: *tuned beats base on Spec adherence AND
  Robustness*.
- An **error-analysis** section: which deterministic checks the tuned model still fails, as the
  starting point for the human paragraph ("is it a data problem?").

## How to run (once you have a trained model)

The model under test is any `StoryGenerator`, so swapping models needs no harness change.

By default the eval runs on **all shipped languages (en, zh, ja)** in one invocation, writing
`results_<lang>.{md,json}` for each — so no language is silently skipped. Pass
`--language en` (or `--language en,zh`) to restrict to a subset.

The judge (Layer 5 rubric) + cloze inferability run **automatically** when `OPENAI_API_KEY` and
`JUDGE_MODEL` are set in `.env`; add `--no-judge` for a deterministic-only run.

```bash
# Offline smoke (mock for every role), all three languages — proves the harness end to end:
python -m islm.eval.run --mock --adversarial

# Local fine-tuned checkpoint (base + LoRA adapter), judge auto-enabled from .env:
python -m islm.eval.run \
    --base-path HuggingFaceTB/SmolLM2-135M-Instruct \
    --tuned-path HuggingFaceTB/SmolLM2-135M-Instruct --tuned-adapter outputs/day3_lora_v3 \
    --adversarial --out evals/results

# Deterministic only (no LLM judge), e.g. a quick regression check:
python -m islm.eval.run --base-path <base> --tuned-path <base> --tuned-adapter <out> --no-judge

# Add --guard to evaluate the deployed system (inference-time validate-and-rewrite), not just
# the raw model.
```

Held-out and adversarial scenarios live at `evals/scenarios/{heldout,adversarial}_<lang>.jsonl`
(committed for reproducibility; auto-created on first run). They are scenario-level distinct from
the training/seed data, so there is no leakage. Results go to `evals/results/` (git-ignored).

### Per-word-list coverage (incl. exam-level English)

Every shipped word list is exercised by an eval so none can silently rot — CEFR (en), HSK (zh),
JLPT (ja), and the **GRE/SAT/ACT exam words** (en). `tests/test_wordlist_evals.py` runs one eval
category per set through the real harness; the English exam category is required and uses the
GRE/SAT/ACT words as the *to-learn* targets. A committed exam scenario set,
`evals/scenarios/heldout_exam_en.jsonl`, lets you evaluate any model on exam-level vocabulary:

```bash
python -m islm.eval.run --language en --scenarios evals/scenarios/heldout_exam_en.jsonl \
    --base-path HuggingFaceTB/SmolLM2-135M-Instruct --out evals/exam
```

## Tracking results over time

A single eval writes a one-off report; to see whether the model is getting **better across runs
and days**, add `--track` (PRD 11's lightweight "CSV/JSONL" tracking option). Each tracked run
appends one line to `evals/runs.jsonl` (an append-only history) and regenerates
`evals/LEADERBOARD.md` (a compact base-vs-tuned table, newest first). Both are committed, so the
progression stays in the repo.

```bash
# Evaluate and record the run in one step (all three languages -> one row each on the board):
python -m islm.eval.run --curated \
    --base-path HuggingFaceTB/SmolLM2-135M-Instruct \
    --tuned-path HuggingFaceTB/SmolLM2-135M-Instruct --tuned-adapter outputs/day3_lora \
    --track --run-label day3-seed-lora --dataset data/curated/seed --epochs 3 --out evals/day3

# Or record a finished results file / rebuild the board from history:
python -m islm.eval.track --results evals/day3/results_en.json --run-label day3-seed-lora \
    --dataset data/curated/seed
python -m islm.eval.track --rebuild
```

Each leaderboard cell is `base→tuned (delta)`; delta = tuned − base (higher is better except OOV,
where lower is better). Blank cells mean "not measured that run" (e.g. the judge is skipped when
no API key is set). The tracker is unit-tested in `tests/test_track.py`.

## Success criteria (PRD 15)

- Tuned **beats prompted base on Spec adherence and Robustness** (primary).
- OOV ≤ 2% (coverage → 100%), ≤1 new word in 100% of sentences, recurrence ≥ 90% — all better than base.
- A reproducible results table + error-analysis paragraph.

## Automated tests for the eval harness

The eval logic itself is unit-tested (`python -m pytest`), so the numbers can be trusted:

| Test (`tests/`) | What it locks down |
| --- | --- |
| `test_eval.py::test_spec_dimensions_match_appendix_a` | Judge dimensions match spec Appendix A |
| `test_eval.py::test_judge_parses_fenced_json_and_clamps` | Judge JSON parsing + 0–2 clamping |
| `test_eval.py::test_judge_handles_garbage` | Malformed judge output → all zeros (no crash) |
| `test_eval.py::test_evaluate_and_aggregate_with_mock` | Harness runs; aggregate has hard + judge metrics |
| `test_eval.py::test_adversarial_targets_not_in_small_known` | Adversarial scenarios are well-formed |
| `test_eval.py::test_error_analysis_counts_failures` | Error analysis tallies the right failure tags |
| `test_eval.py::test_results_markdown_has_sections_and_verdict` | Report has all sections + win verdict |
| `test_eval.py::test_guard_recovers_a_bad_generation` | Inference guard rewrites a failing story |
| `test_validators.py`, `test_multilingual.py` | The deterministic checks the eval relies on |
