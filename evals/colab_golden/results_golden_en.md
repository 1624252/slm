# Eval results: base (Qwen/Qwen3-4B-Instruct-2507) vs tuned (Qwen/Qwen3-4B-Instruct-2507)

Held-out scenarios: **39**.

## Behavioral checks (deterministic — the failures the spec forbids)
| Metric | Base | Tuned | Delta | Better |
| --- | --- | --- | --- | --- |
| Hard-check pass rate (target 1.000) | 0.000 | 0.974 | +0.974 | tuned |
| OOV rate (target <=0.02) | 0.157 | 0.002 | -0.155 | tuned |
| <=1 new word/sentence (target 1.000) | 0.000 | 1.000 | +1.000 | tuned |
| Recurrence satisfied (target 1.000) | 0.462 | 1.000 | +0.538 | tuned |
| Inferability (cloze; target 1.000) | 0.252 | 0.000 | -0.252 | base |

## LLM-as-judge rubric (0-2; first four are spec Appendix A)
| Metric | Base | Tuned | Delta | Better |
| --- | --- | --- | --- | --- |
| spec_adherence (target >=1.5) | 0.872 | 0.923 | +0.051 | tuned |
| robustness (target >=1.5) | 1.026 | 0.821 | -0.205 | base |
| task_quality (target >=1.5) | 1.462 | 0.256 | -1.206 | base |
| consistency (target >=1.5) | 1.692 | 0.795 | -0.897 | base |
| inferability (target >=1.5) | 1.359 | 0.821 | -0.538 | base |
| seductive_detail_control (target >=1.5) | 1.641 | 0.897 | -0.744 | base |
| coherence (target >=1.5) | 1.359 | 0.231 | -1.128 | base |
| interestingness (target >=1.5) | 1.359 | 0.026 | -1.333 | base |

## Win condition (spec)
Beats base on Spec adherence AND Robustness: **FAIL** (spec-adherence up (judge_spec_adherence), robustness not up).

## Error analysis (tuned, held-out)
1/39 outputs failed a check. Most common:
- `oov`: 1
- `coverage`: 1

_Fill in: are the remaining failures a data problem (e.g. under-represented targets, themes that tempt off-vocab words)? What data change would fix them?_
---
_Legend: rates are fractions in [0,1]; judge scores in [0,2] (0 = fails, 1 = partial, 2 = fully). Hard-check pass rate, <=1-new-word, recurrence, coverage: **higher is better** (1.0 = every story passes). OOV rate: **lower is better** (0 = ideal; gate <= 0.02). OOV = out-of-vocabulary (a word not in the learner's known K or target T set); coverage = 1 - OOV rate._
