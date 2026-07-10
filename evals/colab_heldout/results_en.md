# Eval results: base (Qwen/Qwen3-4B-Instruct-2507) vs tuned (Qwen/Qwen3-4B-Instruct-2507)

Held-out scenarios: **12**.

## Behavioral checks (deterministic — the failures the spec forbids)
| Metric | Base | Tuned | Delta | Better |
| --- | --- | --- | --- | --- |
| Hard-check pass rate (target 1.000) | 0.000 | 1.000 | +1.000 | tuned |
| OOV rate (target <=0.02) | 0.092 | 0.001 | -0.091 | tuned |
| <=1 new word/sentence (target 1.000) | 0.000 | 1.000 | +1.000 | tuned |
| Recurrence satisfied (target 1.000) | 0.250 | 1.000 | +0.750 | tuned |
| Inferability (cloze; target 1.000) | 0.083 | 0.042 | -0.041 | base |

## LLM-as-judge rubric (0-2; first four are spec Appendix A)
| Metric | Base | Tuned | Delta | Better |
| --- | --- | --- | --- | --- |
| spec_adherence (target >=1.5) | 0.583 | 1.000 | +0.417 | tuned |
| robustness (target >=1.5) | 0.667 | 1.167 | +0.500 | tuned |
| task_quality (target >=1.5) | 1.083 | 0.417 | -0.666 | base |
| consistency (target >=1.5) | 1.583 | 1.417 | -0.166 | base |
| inferability (target >=1.5) | 1.250 | 1.000 | -0.250 | base |
| seductive_detail_control (target >=1.5) | 1.417 | 1.250 | -0.167 | base |
| coherence (target >=1.5) | 1.000 | 0.417 | -0.583 | base |
| interestingness (target >=1.5) | 0.750 | 0.000 | -0.750 | base |

## Robustness (adversarial set: tiny vocab + jargon themes, n=12)
| Metric | Base | Tuned | Delta | Better |
| --- | --- | --- | --- | --- |
| Adversarial hard-check pass | 0.000 | 0.667 | +0.667 | tuned |

## Win condition (spec)
Beats base on Spec adherence AND Robustness: **PASS** (spec-adherence up (judge_spec_adherence), robustness up).

## Error analysis (tuned, held-out)
All 12 tuned outputs passed the deterministic checks.

_Fill in: are the remaining failures a data problem (e.g. under-represented targets, themes that tempt off-vocab words)? What data change would fix them?_
---
_Legend: rates are fractions in [0,1]; judge scores in [0,2] (0 = fails, 1 = partial, 2 = fully). Hard-check pass rate, <=1-new-word, recurrence, coverage: **higher is better** (1.0 = every story passes). OOV rate: **lower is better** (0 = ideal; gate <= 0.02). OOV = out-of-vocabulary (a word not in the learner's known K or target T set); coverage = 1 - OOV rate._
