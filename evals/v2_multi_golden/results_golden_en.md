# Eval results: base (Qwen/Qwen3-4B-Instruct-2507) vs tuned (Qwen/Qwen3-4B-Instruct-2507)

Held-out scenarios: **39**.

## Behavioral checks (deterministic — the failures the spec forbids)
| Metric | Base | Tuned | Delta | Better |
| --- | --- | --- | --- | --- |
| Hard-check pass rate (target 1.000) | 0.000 | 0.615 | +0.615 | tuned |
| OOV rate (target <=0.02) | 0.157 | 0.016 | -0.141 | tuned |
| <=1 new word/sentence (target 1.000) | 0.000 | 0.692 | +0.692 | tuned |
| Recurrence satisfied (target 1.000) | 0.462 | 0.974 | +0.513 | tuned |
| Inferability (cloze; target 1.000) | 0.252 | 0.372 | +0.120 | tuned |

## LLM-as-judge rubric (0-2; first four are spec Appendix A)
| Metric | Base | Tuned | Delta | Better |
| --- | --- | --- | --- | --- |
| spec_adherence (target >=1.5) | 0.897 | 1.077 | +0.180 | tuned |
| robustness (target >=1.5) | 1.051 | 1.385 | +0.334 | tuned |
| task_quality (target >=1.5) | 1.436 | 1.333 | -0.103 | base |
| consistency (target >=1.5) | 1.718 | 1.692 | -0.026 | base |
| inferability (target >=1.5) | 1.385 | 1.872 | +0.487 | tuned |
| seductive_detail_control (target >=1.5) | 1.615 | 1.872 | +0.257 | tuned |
| coherence (target >=1.5) | 1.359 | 1.205 | -0.154 | base |
| interestingness (target >=1.5) | 1.333 | 1.256 | -0.077 | base |

## Win condition (spec)
Beats base on Spec adherence AND Robustness: **PASS** (spec-adherence up (judge_spec_adherence), robustness up).

## Error analysis (tuned, held-out)
15/39 outputs failed a check. Most common:
- `oov`: 13
- `coverage`: 13
- `one_new_word`: 12
- `recurrence`: 1

_Fill in: are the remaining failures a data problem (e.g. under-represented targets, themes that tempt off-vocab words)? What data change would fix them?_
---
_Legend: rates are fractions in [0,1]; judge scores in [0,2] (0 = fails, 1 = partial, 2 = fully). Hard-check pass rate, <=1-new-word, recurrence, coverage: **higher is better** (1.0 = every story passes). OOV rate: **lower is better** (0 = ideal; gate <= 0.02). OOV = out-of-vocabulary (a word not in the learner's known K or target T set); coverage = 1 - OOV rate._
