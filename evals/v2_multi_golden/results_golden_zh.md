# Eval results: base (Qwen/Qwen3-4B-Instruct-2507) vs tuned (Qwen/Qwen3-4B-Instruct-2507)

Held-out scenarios: **8**.

## Behavioral checks (deterministic — the failures the spec forbids)
| Metric | Base | Tuned | Delta | Better |
| --- | --- | --- | --- | --- |
| Hard-check pass rate (target 1.000) | 0.000 | 0.000 | +0.000 | base |
| OOV rate (target <=0.02) | 0.328 | 0.064 | -0.264 | tuned |
| <=1 new word/sentence (target 1.000) | 0.000 | 0.250 | +0.250 | tuned |
| Recurrence satisfied (target 1.000) | 1.000 | 1.000 | +0.000 | base |
| Inferability (cloze; target 1.000) | 0.625 | 0.250 | -0.375 | base |

## LLM-as-judge rubric (0-2; first four are spec Appendix A)
| Metric | Base | Tuned | Delta | Better |
| --- | --- | --- | --- | --- |
| spec_adherence (target >=1.5) | 0.875 | 1.000 | +0.125 | tuned |
| robustness (target >=1.5) | 1.125 | 1.500 | +0.375 | tuned |
| task_quality (target >=1.5) | 1.500 | 1.000 | -0.500 | base |
| consistency (target >=1.5) | 1.625 | 1.375 | -0.250 | base |
| inferability (target >=1.5) | 1.875 | 1.875 | +0.000 | base |
| seductive_detail_control (target >=1.5) | 1.875 | 1.375 | -0.500 | base |
| coherence (target >=1.5) | 1.375 | 1.000 | -0.375 | base |
| interestingness (target >=1.5) | 1.375 | 0.750 | -0.625 | base |

## Win condition (spec)
Beats base on Spec adherence AND Robustness: **PASS** (spec-adherence up (judge_spec_adherence), robustness up).

## Error analysis (tuned, held-out)
8/8 outputs failed a check. Most common:
- `oov`: 7
- `coverage`: 7
- `one_new_word`: 6

_Fill in: are the remaining failures a data problem (e.g. under-represented targets, themes that tempt off-vocab words)? What data change would fix them?_
---
_Legend: rates are fractions in [0,1]; judge scores in [0,2] (0 = fails, 1 = partial, 2 = fully). Hard-check pass rate, <=1-new-word, recurrence, coverage: **higher is better** (1.0 = every story passes). OOV rate: **lower is better** (0 = ideal; gate <= 0.02). OOV = out-of-vocabulary (a word not in the learner's known K or target T set); coverage = 1 - OOV rate._
