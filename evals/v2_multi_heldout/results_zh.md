# Eval results: base (Qwen/Qwen3-4B-Instruct-2507) vs tuned (Qwen/Qwen3-4B-Instruct-2507)

Held-out scenarios: **12**.

## Behavioral checks (deterministic — the failures the spec forbids)
| Metric | Base | Tuned | Delta | Better |
| --- | --- | --- | --- | --- |
| Hard-check pass rate (target 1.000) | 0.000 | 0.000 | +0.000 | base |
| OOV rate (target <=0.02) | 0.386 | 0.126 | -0.260 | tuned |
| <=1 new word/sentence (target 1.000) | 0.000 | 0.333 | +0.333 | tuned |
| Recurrence satisfied (target 1.000) | 0.917 | 1.000 | +0.083 | tuned |
| Inferability (cloze; target 1.000) | 0.500 | 0.042 | -0.458 | base |

## LLM-as-judge rubric (0-2; first four are spec Appendix A)
| Metric | Base | Tuned | Delta | Better |
| --- | --- | --- | --- | --- |
| spec_adherence (target >=1.5) | 0.583 | 1.000 | +0.417 | tuned |
| robustness (target >=1.5) | 0.833 | 1.083 | +0.250 | tuned |
| task_quality (target >=1.5) | 1.250 | 0.917 | -0.333 | base |
| consistency (target >=1.5) | 1.333 | 1.667 | +0.334 | tuned |
| inferability (target >=1.5) | 1.583 | 1.750 | +0.167 | tuned |
| seductive_detail_control (target >=1.5) | 1.583 | 1.500 | -0.083 | base |
| coherence (target >=1.5) | 1.083 | 0.750 | -0.333 | base |
| interestingness (target >=1.5) | 1.333 | 0.917 | -0.416 | base |

## Robustness (adversarial set: tiny vocab + jargon themes, n=12)
| Metric | Base | Tuned | Delta | Better |
| --- | --- | --- | --- | --- |
| Adversarial hard-check pass | 0.000 | 0.000 | +0.000 | base |

## Win condition (spec)
Beats base on Spec adherence AND Robustness: **FAIL** (spec-adherence up (judge_spec_adherence), robustness not up).

## Error analysis (tuned, held-out)
12/12 outputs failed a check. Most common:
- `oov`: 12
- `coverage`: 12
- `one_new_word`: 8

_Fill in: are the remaining failures a data problem (e.g. under-represented targets, themes that tempt off-vocab words)? What data change would fix them?_
---
_Legend: rates are fractions in [0,1]; judge scores in [0,2] (0 = fails, 1 = partial, 2 = fully). Hard-check pass rate, <=1-new-word, recurrence, coverage: **higher is better** (1.0 = every story passes). OOV rate: **lower is better** (0 = ideal; gate <= 0.02). OOV = out-of-vocabulary (a word not in the learner's known K or target T set); coverage = 1 - OOV rate._
