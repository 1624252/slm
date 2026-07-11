# Eval results: base (Qwen/Qwen3-4B-Instruct-2507) vs tuned (Qwen/Qwen3-4B-Instruct-2507)

Held-out scenarios: **12**.

## Behavioral checks (deterministic — the failures the spec forbids)
| Metric | Base | Tuned | Delta | Better |
| --- | --- | --- | --- | --- |
| Hard-check pass rate (target 1.000) | 0.000 | 0.333 | +0.333 | tuned |
| OOV rate (target <=0.02) | 0.392 | 0.046 | -0.346 | tuned |
| <=1 new word/sentence (target 1.000) | 0.000 | 0.750 | +0.750 | tuned |
| Recurrence satisfied (target 1.000) | 0.750 | 1.000 | +0.250 | tuned |
| Inferability (cloze; target 1.000) | 0.208 | 0.000 | -0.208 | base |

## LLM-as-judge rubric (0-2; first four are spec Appendix A)
| Metric | Base | Tuned | Delta | Better |
| --- | --- | --- | --- | --- |
| spec_adherence (target >=1.5) | 0.583 | 0.833 | +0.250 | tuned |
| robustness (target >=1.5) | 0.917 | 1.000 | +0.083 | tuned |
| task_quality (target >=1.5) | 1.083 | 0.583 | -0.500 | base |
| consistency (target >=1.5) | 1.417 | 1.000 | -0.417 | base |
| inferability (target >=1.5) | 1.833 | 1.167 | -0.666 | base |
| seductive_detail_control (target >=1.5) | 1.750 | 1.167 | -0.583 | base |
| coherence (target >=1.5) | 1.000 | 0.417 | -0.583 | base |
| interestingness (target >=1.5) | 1.167 | 0.250 | -0.917 | base |

## Robustness (adversarial set: tiny vocab + jargon themes, n=12)
| Metric | Base | Tuned | Delta | Better |
| --- | --- | --- | --- | --- |
| Adversarial hard-check pass | 0.000 | 0.083 | +0.083 | tuned |

## Win condition (spec)
Beats base on Spec adherence AND Robustness: **PASS** (spec-adherence up (judge_spec_adherence), robustness up).

## Error analysis (tuned, held-out)
8/12 outputs failed a check. Most common:
- `oov`: 8
- `coverage`: 8
- `one_new_word`: 3

_Fill in: are the remaining failures a data problem (e.g. under-represented targets, themes that tempt off-vocab words)? What data change would fix them?_
---
_Legend: rates are fractions in [0,1]; judge scores in [0,2] (0 = fails, 1 = partial, 2 = fully). Hard-check pass rate, <=1-new-word, recurrence, coverage: **higher is better** (1.0 = every story passes). OOV rate: **lower is better** (0 = ideal; gate <= 0.02). OOV = out-of-vocabulary (a word not in the learner's known K or target T set); coverage = 1 - OOV rate._
