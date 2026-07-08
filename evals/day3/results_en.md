# Eval results: base (HuggingFaceTB/SmolLM2-135M-Instruct) vs tuned (HuggingFaceTB/SmolLM2-135M-Instruct)

Held-out scenarios: **8**.

## Behavioral checks (deterministic — the failures the spec forbids)
| Metric | Base | Tuned | Delta | Better |
| --- | --- | --- | --- | --- |
| Hard-check pass rate | 0.000 | 0.000 | +0.000 | base |
| OOV rate | 0.413 | 0.170 | -0.243 | tuned |
| <=1 new word/sentence | 0.000 | 0.000 | +0.000 | base |
| Recurrence satisfied | 0.125 | 0.125 | +0.000 | base |

## LLM-as-judge rubric (0-2; first four are spec Appendix A)
| Metric | Base | Tuned | Delta | Better |
| --- | --- | --- | --- | --- |

## Win condition (spec)
Beats base on Spec adherence AND Robustness: **FAIL** (spec-adherence not up (hard_pass_rate), robustness not up).

## Error analysis (tuned, held-out)
8/8 outputs failed a check. Most common:
- `oov`: 8
- `coverage`: 8
- `one_new_word`: 8
- `recurrence`: 7

_Fill in: are the remaining failures a data problem (e.g. under-represented targets, themes that tempt off-vocab words)? What data change would fix them?_
---
_Legend: rates are fractions in [0,1]; judge scores in [0,2] (0 = fails, 1 = partial, 2 = fully). Hard-check pass rate, <=1-new-word, recurrence, coverage: **higher is better** (1.0 = every story passes). OOV rate: **lower is better** (0 = ideal; gate <= 0.02). OOV = out-of-vocabulary (a word not in the learner's known K or target T set); coverage = 1 - OOV rate._
