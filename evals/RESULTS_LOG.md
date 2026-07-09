# Results log (for future analysis)

A narrative companion to the machine-readable `runs.jsonl` / `LEADERBOARD.md`. Each entry records
**what we changed, why, and what the eval did** — the context a bare metrics row can't hold. Newest
first. The GPU/Colab QLoRA run (deferred) will append here too.

Metrics are the deterministic (non-AI) validator checks, base vs. tuned, on held-out scenarios:
`hard_pass` (all checks), `OOV` (out-of-vocabulary rate, lower better), `<=1-new` (≤1 new word per
sentence), `recurrence` (target repeated ≥3×). See `docs/EVALUATION.md`.

---

## Root-cause finding (2026-07-08): the prompt didn't fit the training window

**Symptom.** Through Day 2 and the first Day 3, tuned `hard_pass` never moved off 0.000; only OOV
drifted a little. Looked like "tiny model, too few steps." It wasn't.

**Actual cause — a data bug, not a compute one.** Each training record's prompt embedded the
**full 2291-word baseline** as `KNOWN_WORDS`, rendering records to **5k–12k tokens**. The assistant
story (the completion we train on) sits at the *end*. Training left-truncated to 256 tokens to keep
the story — but `TARGET_WORDS`, the theme, and the rules all sit *before* the giant word list, so
truncation **deleted the entire task description**. The model trained on
`"[tail of a word list] Write 8-14 sentences. [story]"` and never saw what it was supposed to do.
Meanwhile the eval prompt used only the 146-word curated list — so train and eval didn't even match.

**Fix (in data, per the Day-4 "fix in data not hyperparameters" principle).** `seed.py` now scopes
each record's `KNOWN_WORDS` to **the curated-small baseline (~146) ∪ the story's own content
words**, minus the target. Records dropped to **539–704 tokens (mean 610)** — the whole prompt now
fits a 768-token window uncut, and matches the eval's `--curated` setup.

| | Before | After |
| --- | --- | --- |
| KNOWN_WORDS per record | 2291 words | ~150 words |
| Record size | 5k–12k tokens | 539–704 tokens |
| Task text (TARGET_WORDS + rules) survives truncation? | **No** | **Yes** |
| Train/eval prompt match | No (2291 vs 146) | Yes (both ~146 curated) |

---

## Runs

<!-- newest first; append a block per run -->

### 2026-07-09 — `v7` (4× teacher data) — under-trained on CPU; v5 still best

**The data-scaling test.** Only variable vs v5: the dataset. Used the teacher endpoint to generate
a **4× larger corpus** — `data/v7_corpus`, **87 train records** (en 50 / zh 30 / ja 7) = compact
teacher(en/zh/ja) + the 22-record seed. Same hyperparameters as v5 (r32/α64, lr2e-4, seq 1024)
**except** steps: an overnight CPU slowdown (~150 s/step) forced `--max-steps 40` to fit the budget
(v5 ran 110). Final train loss **1.99** (v5: 0.87) — i.e. **under-trained**.

Golden set, tuned v5 → v7 (all languages):

| lang | OOV | hard-pass | ≤1-new | coherence | judge overall |
| --- | --- | --- | --- | --- | --- |
| en | 0.133 → 0.222 | 0.103 → **0.000** | 0.487 → **0.026** | 0.15 → 0.20 | 0.45 → **0.16** |
| zh | 0.250 → 0.359 | 0.000 → **0.125** | 0.125 → **0.500** | 0.00 → 0.00 | 0.19 → 0.03 |
| ja | 0.117 → **0.612** | 0.250 → **0.000** | 0.500 → 0.375 | 0.00 → 0.00 | 0.05 → 0.00 |

(zh is mixed — OOV worse but hard-pass and ≤1-new actually *up*, a hint that more zh data helped
even at 40 steps. en and ja clearly worse — ja worst hit (OOV 0.12→0.61), unsurprising given its
tiny 7-record share under only 40 steps. Net: v7 does not beat v5, but the zh signal supports the
"data helps, just needs GPU steps" reading.)

**Verdict: v7 does NOT beat v5** — worse OOV, lost hard-passes, ≤1-new collapsed. But this is a
**confounded test**, not evidence against data scaling: 40 steps (loss 1.99) is far too few for the
model to fit the constraints, regardless of data quality. On CPU, more data needs more steps, and
the box can't afford both. **What we actually learned:** the data-scaling hypothesis is *untestable
on CPU* — it needs the GPU (4× data + adequate steps together). v5 remains the best adapter.

---

## FINAL OVERNIGHT SUMMARY (2026-07-09, ~07:00)

**Best model: `v5`** (`outputs/day3_lora_v5`) — unchanged by the overnight work, but now far better
understood. Golden set, base→tuned: en OOV 0.43→0.13, zh 0.88→0.25, ja 1.00→0.12; en/ja post
hard-passes; spec-adherence/robustness/consistency up. Its one weakness — coherence/interestingness
drop to ~0 — is the open problem.

**The v3→v7 arc and what it proved:**
- v3→v5: hyperparameter search (epochs, rank, LR) improved OOV but plateaued; every tuned model
  traded prose quality (coherence/interestingness) for vocabulary control.
- **v6** (epochs 3): NEGATIVE — worse than v5 on golden. → hyperparameter tuning is **capped** for a
  135M model on 22 examples.
- **v7** (4× teacher data): under-trained on CPU (40 steps), lost to v5. → data scaling is the right
  lever but is **compute-bound on CPU** — can't get more data *and* enough steps in one night.
- **Both roads converge on the same conclusion: the real gains need the Colab GPU run.** A 4B model
  (Qwen3-4B) with QLoRA can hold the constraints without flattening the story AND absorb a larger
  teacher corpus with adequate steps — exactly what CPU cannot.

**Durable overnight artifacts (all committed):**
- Teacher data pipeline made **resilient** (client retries + skip-failed-scenario) after a network
  blip killed a batch — critical for the unattended GPU corpus run.
- `islm.datagen.compact` — fixes the teacher records' giant-prompt bug (~5.2k→~830 tok); **required
  before training on any teacher data**.
- A validated teacher corpus recipe: en 60% / zh 75% / ja 10% keep rates measured.
- v6 + v7 fully evaluated (all criteria) and documented.

**Next step (for the GPU):** `docs/COLAB_PLAN.md` — generate a few-thousand-record teacher corpus
(compact it), QLoRA-tune Qwen3-4B with the aligned recipe + adequate steps, eval on golden. That is
the run expected to beat v5 on both OOV *and* coherence/interestingness.

### 2026-07-09 — `v6` (epochs 3 vs v5's 5) — NEGATIVE RESULT, v5 still best

**One variable changed** from v5: epochs 5→3 (r32/α64, lr2e-4, aligned recipe). Hypothesis: the
coherence/interestingness collapse is over-fitting, so less training should preserve prose quality.
Final train loss 1.27 (v5: 0.87). Judge = claude-sonnet-4-6.

Golden set, base→tuned (the canonical target):

| lang | OOV (v6) | OOV (v5) | hard (v6/v5) | coherence (v6/v5) |
| --- | --- | --- | --- | --- |
| en | 0.425→**0.127** | →0.133 | 0.051 / **0.103** | 0.23 / 0.15 |
| zh | 0.881→0.377 | →**0.250** | 0.125 / 0.000 | 0.00 / 0.00 |
| ja | 1.000→0.480 | →**0.117** | 0.000 / **0.250** | 0.00 / 0.00 |

**Verdict: v6 does NOT beat v5.** Fewer epochs nudged English coherence up (0.15→0.23) but
**regressed OOV badly on zh (0.25→0.38) and ja (0.12→0.48)** and lost ja's hard-passes. Held-out
was roughly comparable to v5, but golden — the correctness target — is clearly worse. Interpretation:
the quality collapse is a **model-capacity limit, not over-fitting**; epoch count isn't the lever.
This is why the overnight plan pivots to **scaling the dataset** (v7): a 135M model on 22 examples
is near its ceiling regardless of hyperparameters. v5 remains the best adapter.

### 2026-07-08 — `v5` (aligned QLoRA recipe; **judged**, all criteria, **on the golden set**)

First run scored on **all three criteria families** — deterministic checks, the 8-dimension
LLM-judge rubric (judge = `claude-sonnet-4-6`), and cloze inferability — and the first run
evaluated **on the golden set** (Layer 1), not just held-out. Base and tuned both judged on the
same inputs.

**Hyperparameters** (`outputs/day3_lora_v5/train_summary.json`): SmolLM2-135M-Instruct, LoRA
r=32/α=64, **5 epochs / 110 steps**, lr 2e-4, seq 768, **cosine schedule + 3% warmup** (aligned
QLoRA recipe), adamw_torch (CPU), final train loss **0.87**.

Deterministic + cloze, base→tuned:

| Target / lang | Hard-pass (→1) | OOV (→≤.02) | ≤1-new (→1) | Recurrence (→1) | Cloze infer. |
| --- | --- | --- | --- | --- | --- |
| golden en | 0.000→**0.103** | 0.425→**0.133** | 0.000→**0.487** | 0.205→**0.385** | 0.179→0.000 |
| golden zh | 0.000→0.000 | 0.881→**0.250** | 0.000→0.125 | 0.000→**0.875** | 0.000→0.125 |
| golden ja | 0.000→**0.250** | 1.000→**0.117** | 0.000→**0.500** | 0.000→**0.625** | 0.000→0.125 |
| heldout en | 0.000→**0.125** | 0.413→**0.080** | 0.000→**0.500** | 0.125→**0.250** | 0.250→0.000 |
| heldout zh | 0.000→0.000 | 0.856→**0.245** | 0.000→0.125 | 0.125→**0.625** | 0.000→0.062 |
| heldout ja | 0.000→**0.125** | 1.000→**0.142** | 0.000→**0.375** | 0.000→**0.375** | 0.000→0.062 |
| exam en | 0.000→**0.125** | 0.430→**0.155** | 0.000→**0.500** | 0.000→0.125 | 0.000→0.000 |

Judge rubric (0–2), base→tuned, key dimensions:

| Target / lang | spec_adh | robustness | consistency | coherence | interestingness | overall |
| --- | --- | --- | --- | --- | --- | --- |
| golden en | 0.00→**0.59** | 0.00→**0.51** | 0.10→**1.00** | 0.23→0.15 | 0.05→0.00 | 0.14→**0.45** |
| golden zh | — | — | — | 0.50→0.00 | 0.25→0.00 | 0.16→0.19 |
| golden ja | — | — | — | 0.88→0.00 | 0.12→0.00 | 0.19→0.05 |
| heldout en | 0.00→**0.5+** | up | up | 0.38→0.00 | 0.12→0.00 | 0.20→**0.31** |
| exam en | up | up | up | 0.25→**0.38** | 0.00→0.00 | 0.06→**0.33** |

**Read.** The deterministic + spec-judge story is strongly positive everywhere: OOV collapses
(ja 1.00→0.12, zh 0.88→0.25, en 0.43→0.13 on golden), and en/ja post hard-passes on both golden
and held-out. Judge spec_adherence/robustness/consistency all rise. **But every quality
dimension — coherence, interestingness, task_quality — drops, often to 0**, most starkly on
zh/ja golden (coherence 0.88→0.00 for ja). This is the real finding the new criteria surface: the
135M model buys vocabulary control by producing rigid, repetitive prose. Base-model judged scores
are near-zero across the board (the base ignores the word list *and* isn't compelling). Win
condition (spec_adherence AND robustness up): **PASS** on en. The quality regression is the
Colab-scale lever: a 4B model should hold the constraints without flattening the story.

**Iterations & hyperparameters** (from `outputs/day3_lora_v3/train_summary.json`):

| Setting | Value | | Setting | Value |
| --- | --- | --- | --- | --- |
| Base model | `SmolLM2-135M-Instruct` | | LoRA rank `r` | **32** (v2: 16) |
| Method | LoRA (CPU) | | LoRA `alpha` | **64** (v2: 32) |
| **Epochs** | **5** (v2: 3) | | LoRA `dropout` | 0.05 |
| **Optimizer steps** | **110** (v2: 66) | | Learning rate | 2e-4 |
| Batch / grad-accum | 1 / 1 | | Max sequence length | 768 |
| Final train loss | **0.86** (v2: 1.63) | | Seed | 0 |

**Change from v2:** 5 epochs (was 3) and double the LoRA capacity (r=32/α=64 vs 16/32). Rationale:
on 22 examples the cheap levers are more passes + more adapter capacity to absorb the constrained
style.

| Lang | OOV (base→tuned) | Hard-pass (→1.000) | ≤1-new (→1.000) | Recurrence (→1.000) | vs v2 |
| --- | --- | --- | --- | --- | --- |
| en | 0.413 → **0.096** (−0.317) | 0.000 → **0.125** | 0.000 → **0.500** | 0.125 → 0.125 | better (v2 OOV 0.170, hard 0) |
| zh | 0.856 → **0.279** (−0.577) | 0.000 → 0.000 | 0.000 → 0.000 | 0.125 → **0.875** | mixed (better OOV/recur; lost v2's hard 0.25) |
| ja | 1.000 → **0.167** (−0.833) | 0.000 → **0.250** | 0.000 → **0.625** | 0.000 → **0.500** | far better (v2 OOV 0.874) |

**Read.** v3 beats v2 clearly: English and Japanese post their first hard-passes, OOV drops hard
across all three languages, and ≤1-new-word/recurrence move up substantially. The one regression is
zh hard-pass (0.25 → 0), even though zh OOV and recurrence improved — the tighter fit helped most
checks but pushed a couple of zh stories just over a gate. Net: the highest-leverage cheap knobs
(epochs + LoRA capacity) worked. Win condition still FAIL (OOV ≤ 0.02 gate needs the GPU run).

**Exam set** (`day3-v3-exam`, GRE/SAT/ACT targets, `heldout_exam_en`): OOV 0.430 → **0.169**,
≤1-new 0.000 → **0.500**, recurrence 0.000 → 0.125, hard-pass 0.000 → 0.000. The v3 adapter
generalizes to hard exam-vocabulary targets it never trained on — OOV more than halves — even
though no story clears every gate yet.

### 2026-07-08 — `exam-baseline` (GRE/SAT/ACT targets, prompted base model)

**What.** Base `SmolLM2-135M-Instruct` (prompted, no fine-tune) on 8 held-out English scenarios
whose to-learn targets are **GRE/SAT/ACT exam words** (`evals/scenarios/heldout_exam_en.jsonl`),
temperature 0, `--max-new-tokens 200`.

| Metric | Value | Target |
| --- | --- | --- |
| Hard-pass | 0.000 | 1.000 |
| OOV | 0.430 | ≤0.02 |
| ≤1-new | 0.000 | 1.000 |
| Recurrence | 0.000 | 1.000 |

**Read.** The base model fails exam-vocabulary scenarios about the same as CEFR ones (OOV 0.430 vs
0.413) — it ignores the allowed-word list either way. This is the **baseline to beat** once the
harder targets flow into training. It confirms the exam words load, sample as targets, and run
through the real harness end to end. Reproduce: `docs/EVALUATION.md` → "Per-word-list coverage".

### 2026-07-08 — `day3-seed-lora-v2` (the data fix lands)

**Iterations & hyperparameters** (from `outputs/day3_lora/train_summary.json`):

| Setting | Value | | Setting | Value |
| --- | --- | --- | --- | --- |
| Base model | `SmolLM2-135M-Instruct` | | LoRA rank `r` | 16 |
| Method | LoRA (not QLoRA; CPU) | | LoRA `alpha` | 32 |
| Device | CPU | | LoRA `dropout` | 0.05 |
| Train examples | 22 | | `target_modules` | all-linear |
| **Epochs** | **3** | | Learning rate | 2e-4 |
| **Optimizer steps (iterations)** | **66** | | LR schedule | linear (TRL default) |
| Per-device batch size | 1 | | Optimizer | adamw_torch (TRL default) |
| Gradient accumulation | 1 | | Seed | 0 |
| Effective batch size | 1 | | Max sequence length | 768 (left-truncated) |
| Final train loss | **1.63** | | (was 2.33 when records were truncated) | |

Eval: 8 held-out `--curated` scenarios per language, temperature 0, `--max-new-tokens 220`.
Reproduce command is in `docs/DAY3.md` → Reproduce.

Targets in parentheses are the Behavior-Spec thresholds (`config.Thresholds`).

| Lang | OOV (→ ≤0.02) | Hard-pass (→ 1.000) | ≤1-new (→ 1.000) | Recurrence (→ 1.000) |
| --- | --- | --- | --- | --- |
| en | 0.413 → **0.170** (−0.243) | 0.000 → 0.000 | 0.000 → 0.000 | 0.125 → 0.125 |
| zh | 0.856 → **0.439** (−0.417) | 0.000 → **0.250** | 0.000 → **0.250** | 0.125 → **0.375** |
| ja | 1.000 → **0.874** (−0.126) | 0.000 → 0.000 | 0.000 → **0.125** | 0.000 → 0.000 |

**Read.** The eval improved on every language. OOV fell **12×** more than the old truncated run
(en −0.243 vs −0.020), and **zh produced the project's first non-zero hard-pass (0.25)** — direct
evidence the model now sees `TARGET_WORDS` + the rules during training. `hard_pass` stays low overall
because the gate is OOV ≤ 0.02, which a 135M model on CPU LoRA won't clear; that ceiling is a
model/compute limit, not the data bug we just fixed.

**Win condition:** still FAIL (spec bar unchanged) — expected pre-GPU.

**Next levers (need Colab GPU / teacher key, deferred):** (1) a teacher-generated corpus for volume
beyond the 28-story seed; (2) a real multi-epoch **QLoRA** run on a larger base (e.g. Qwen3-4B);
(3) ja is the laggard (OOV still 0.87) — more ja seed stories + a JLPT-scoped known list.
