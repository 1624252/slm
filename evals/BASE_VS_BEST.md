# Base vs. best — final comparison

The headline result: **base model vs. the best fine-tuned adapter**, on the same held-out scenarios,
for the behavior this project instills. Base columns are filled; **tuned columns are placeholders
(`___`) pending the final training run** — I'll drop the numbers in from the committed eval JSON.

- **Base:** `Qwen/Qwen3-4B-Instruct-2507`, prompted, no fine-tune.
- **Best:** the multilingual v2 adapter (fresh-from-base QLoRA on `data/dataset_v2`, 1610 stories,
  widened en/zh/ja palettes). Same base, same prompt — only the training data differs.
- **Judge:** `claude-sonnet-4-6`. Scenarios: golden set (Layer 1) + held-out (Layer 2, incl. adversarial).

---

## 1. The behavior metric (LLM-as-judge, 0–2) — this is the win condition

Per the spec (Appendix A): **a tuned model that beats base on Spec adherence and Robustness is a
win.** Mean judge score per dimension, base → best, same scenarios.

### Golden set

| lang | spec_adherence | robustness | task_quality | consistency | coherence | interestingness | overall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| en | 0.90 → **___** | 1.05 → **___** | 1.44 → ___ | 1.72 → ___ | 1.36 → ___ | 1.33 → ___ | 1.35 → ___ |
| zh | 0.88 → **___** | 1.12 → **___** | 1.50 → ___ | 1.62 → ___ | 1.38 → ___ | 1.38 → ___ | 1.45 → ___ |
| ja | 0.50 → **___** | 0.75 → **___** | 1.00 → ___ | 1.50 → ___ | 1.12 → ___ | 0.88 → ___ | 1.16 → ___ |

### Held-out set (+ adversarial)

| lang | spec_adherence | robustness | task_quality | consistency | coherence | interestingness | overall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| en | 0.58 → **___** | 0.75 → **___** | 1.08 → ___ | 1.67 → ___ | 1.08 → ___ | 0.83 → ___ | 1.08 → ___ |
| zh | 0.58 → **___** | 0.83 → **___** | 1.25 → ___ | 1.33 → ___ | 1.08 → ___ | 1.33 → ___ | 1.20 → ___ |
| ja | 0.58 → **___** | 0.83 → **___** | 1.00 → ___ | 1.42 → ___ | 1.00 → ___ | 1.00 → ___ | 1.15 → ___ |

**Win check (spec_adherence AND robustness up):** en ___ · zh ___ · ja ___

---

## 2. Deterministic checks — the falsifiable backing for the same spec

The behavior metric above is graded by an LLM; these are the mechanical, non-AI checks of the exact
same Behavior Spec. Base → best.

### Golden set

| lang | hard-pass (→1.0) | OOV (→≤0.02) | ≤1-new-word (→1.0) | recurrence (→1.0) |
| --- | --- | --- | --- | --- |
| en | 0.000 → **___** | 0.157 → **___** | 0.000 → **___** | 0.462 → ___ |
| zh | 0.000 → **___** | 0.328 → **___** | 0.000 → **___** | 1.000 → ___ |
| ja | 0.000 → **___** | 0.309 → **___** | 0.000 → **___** | 0.750 → ___ |

### Held-out set (+ adversarial)

| lang | hard-pass (→1.0) | OOV (→≤0.02) | ≤1-new-word (→1.0) | recurrence (→1.0) |
| --- | --- | --- | --- | --- |
| en | 0.000 → **___** | 0.092 → **___** | 0.000 → **___** | 0.250 → ___ |
| zh | 0.000 → **___** | 0.386 → **___** | 0.000 → **___** | 0.917 → ___ |
| ja | 0.000 → **___** | 0.392 → **___** | 0.000 → **___** | 0.750 → ___ |

---

## 3. Error analysis (spec-required)

_To fill after the run: where does the best model still fail, and is it a data problem?_

- **en:** ___
- **zh:** ___ (expected: OOV collapses but may stay above the strict 0.02 gate — jieba segmentation
  compounds like `山上`/`水里`; a known limitation, not a model failure).
- **ja:** ___ (watch prose quality — coherence/interestingness — vs. the constraint gains).
- **Overall data-vs-model read:** ___

---

## How to fill this in

From the committed eval JSON of the final run (base + tuned are in the same files):

```
evals/<run>_golden/results_golden_{en,zh,ja}.json   -> tuned block
evals/<run>_heldout/results_{en,zh,ja}.json         -> tuned block
```

Each `tuned` block has: `hard_pass_rate`, `mean_oov_rate`, `one_new_word_pass_rate`,
`recurrence_pass_rate`, and `judge_{spec_adherence,robustness,task_quality,consistency,coherence,
interestingness,overall}`. Drop those into the `___` cells.
