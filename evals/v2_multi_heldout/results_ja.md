# Eval results: base (Qwen/Qwen3-4B-Instruct-2507) vs tuned (Qwen/Qwen3-4B-Instruct-2507)

Held-out scenarios: **12**.

## Behavioral checks (deterministic — the failures the spec forbids)
| Metric | Base | Tuned | Delta | Better |
| --- | --- | --- | --- | --- |
| Hard-check pass rate (target 1.000) | 0.000 | 0.083 | +0.083 | tuned |
| OOV rate (target <=0.02) | 0.392 | 0.074 | -0.318 | tuned |
| <=1 new word/sentence (target 1.000) | 0.000 | 0.750 | +0.750 | tuned |
| Recurrence satisfied (target 1.000) | 0.750 | 0.833 | +0.083 | tuned |
| Inferability (cloze; target 1.000) | 0.167 | 0.000 | -0.167 | base |

## LLM-as-judge rubric (0-2; first four are spec Appendix A)
| Metric | Base | Tuned | Delta | Better |
| --- | --- | --- | --- | --- |
| spec_adherence (target >=1.5) | 0.583 | 0.583 | +0.000 | base |
| robustness (target >=1.5) | 0.917 | 0.833 | -0.084 | base |
| task_quality (target >=1.5) | 1.083 | 0.667 | -0.416 | base |
| consistency (target >=1.5) | 1.417 | 0.917 | -0.500 | base |
| inferability (target >=1.5) | 1.750 | 1.417 | -0.333 | base |
| seductive_detail_control (target >=1.5) | 1.750 | 1.250 | -0.500 | base |
| coherence (target >=1.5) | 0.833 | 0.500 | -0.333 | base |
| interestingness (target >=1.5) | 1.167 | 0.583 | -0.584 | base |

## Robustness (adversarial set: tiny vocab + jargon themes, n=12)
| Metric | Base | Tuned | Delta | Better |
| --- | --- | --- | --- | --- |
| Adversarial hard-check pass | 0.000 | 0.000 | +0.000 | base |

## Win condition (spec)
Beats base on Spec adherence AND Robustness: **FAIL** (spec-adherence not up (judge_spec_adherence), robustness not up).

## Error analysis (tuned, held-out)
11/12 outputs failed a check. Most common:
- `oov`: 10
- `coverage`: 10
- `one_new_word`: 3
- `recurrence`: 2

_Fill in: are the remaining failures a data problem (e.g. under-represented targets, themes that tempt off-vocab words)? What data change would fix them?_
---
_Legend: rates are fractions in [0,1]; judge scores in [0,2] (0 = fails, 1 = partial, 2 = fully). Hard-check pass rate, <=1-new-word, recurrence, coverage: **higher is better** (1.0 = every story passes). OOV rate: **lower is better** (0 = ideal; gate <= 0.02). OOV = out-of-vocabulary (a word not in the learner's known K or target T set); coverage = 1 - OOV rate._
