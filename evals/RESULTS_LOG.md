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
