# Eval results: base (Qwen/Qwen3-4B-Instruct-2507) vs tuned (Qwen/Qwen3-4B-Instruct-2507)

Held-out scenarios: **39**.

## Behavioral checks (deterministic — the failures the spec forbids)
| Metric | Base | Tuned | Delta | Better |
| --- | --- | --- | --- | --- |
| Hard-check pass rate (target 1.000) | 0.000 | 0.487 | +0.487 | tuned |
| OOV rate (target <=0.02) | 0.157 | 0.018 | -0.139 | tuned |
| <=1 new word/sentence (target 1.000) | 0.000 | 0.564 | +0.564 | tuned |
| Recurrence satisfied (target 1.000) | 0.462 | 0.974 | +0.513 | tuned |
| Inferability (cloze; target 1.000) | 0.278 | 0.355 | +0.077 | tuned |

## LLM-as-judge rubric (0-2; first four are spec Appendix A)
| Metric | Base | Tuned | Delta | Better |
| --- | --- | --- | --- | --- |
| spec_adherence (target >=1.5) | 0.923 | 1.103 | +0.180 | tuned |
| robustness (target >=1.5) | 1.128 | 1.308 | +0.180 | tuned |
| task_quality (target >=1.5) | 1.385 | 1.333 | -0.052 | base |
| consistency (target >=1.5) | 1.795 | 1.692 | -0.103 | base |
| inferability (target >=1.5) | 1.436 | 1.846 | +0.410 | tuned |
| seductive_detail_control (target >=1.5) | 1.692 | 1.744 | +0.052 | tuned |
| coherence (target >=1.5) | 1.282 | 1.051 | -0.231 | base |
| interestingness (target >=1.5) | 1.256 | 1.308 | +0.052 | tuned |

## Win condition (spec)
Beats base on Spec adherence AND Robustness: **PASS** (spec-adherence up (judge_spec_adherence), robustness up).

## Error analysis (tuned, held-out)
20/39 outputs failed a check. Most common:
- `one_new_word`: 17
- `oov`: 15
- `coverage`: 15
- `recurrence`: 1

_Fill in: are the remaining failures a data problem (e.g. under-represented targets, themes that tempt off-vocab words)? What data change would fix them?_
---
_Legend: rates are fractions in [0,1]; judge scores in [0,2] (0 = fails, 1 = partial, 2 = fully). Hard-check pass rate, <=1-new-word, recurrence, coverage: **higher is better** (1.0 = every story passes). OOV rate: **lower is better** (0 = ideal; gate <= 0.02). OOV = out-of-vocabulary (a word not in the learner's known K or target T set); coverage = 1 - OOV rate._
