# Eval results: base (Qwen/Qwen3-4B-Instruct-2507) vs tuned (Qwen/Qwen3-4B-Instruct-2507)

Held-out scenarios: **12**.

## Behavioral checks (deterministic — the failures the spec forbids)
| Metric | Base | Tuned | Delta | Better |
| --- | --- | --- | --- | --- |
| Hard-check pass rate (target 1.000) | 0.000 | 0.417 | +0.417 | tuned |
| OOV rate (target <=0.02) | 0.092 | 0.013 | -0.079 | tuned |
| <=1 new word/sentence (target 1.000) | 0.000 | 0.500 | +0.500 | tuned |
| Recurrence satisfied (target 1.000) | 0.250 | 0.917 | +0.667 | tuned |
| Inferability (cloze; target 1.000) | 0.167 | 0.375 | +0.208 | tuned |

## LLM-as-judge rubric (0-2; first four are spec Appendix A)
| Metric | Base | Tuned | Delta | Better |
| --- | --- | --- | --- | --- |
| spec_adherence (target >=1.5) | 0.583 | 0.833 | +0.250 | tuned |
| robustness (target >=1.5) | 0.750 | 1.250 | +0.500 | tuned |
| task_quality (target >=1.5) | 1.083 | 1.500 | +0.417 | tuned |
| consistency (target >=1.5) | 1.667 | 1.667 | +0.000 | base |
| inferability (target >=1.5) | 1.250 | 1.833 | +0.583 | tuned |
| seductive_detail_control (target >=1.5) | 1.417 | 1.750 | +0.333 | tuned |
| coherence (target >=1.5) | 1.083 | 1.167 | +0.084 | tuned |
| interestingness (target >=1.5) | 0.833 | 1.417 | +0.584 | tuned |

## Robustness (adversarial set: tiny vocab + jargon themes, n=12)
| Metric | Base | Tuned | Delta | Better |
| --- | --- | --- | --- | --- |
| Adversarial hard-check pass | 0.000 | 0.000 | +0.000 | base |

## Win condition (spec)
Beats base on Spec adherence AND Robustness: **FAIL** (spec-adherence up (judge_spec_adherence), robustness not up).

## Error analysis (tuned, held-out)
7/12 outputs failed a check. Most common:
- `one_new_word`: 6
- `oov`: 2
- `coverage`: 2
- `recurrence`: 1

_Fill in: are the remaining failures a data problem (e.g. under-represented targets, themes that tempt off-vocab words)? What data change would fix them?_
---
_Legend: rates are fractions in [0,1]; judge scores in [0,2] (0 = fails, 1 = partial, 2 = fully). Hard-check pass rate, <=1-new-word, recurrence, coverage: **higher is better** (1.0 = every story passes). OOV rate: **lower is better** (0 = ideal; gate <= 0.02). OOV = out-of-vocabulary (a word not in the learner's known K or target T set); coverage = 1 - OOV rate._
