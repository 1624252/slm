# Eval results: base (Qwen/Qwen3-4B-Instruct-2507) vs tuned (Qwen/Qwen3-4B-Instruct-2507)

Held-out scenarios: **39**.

## Behavioral checks (deterministic — the failures the spec forbids)
| Metric | Base | Tuned | Delta | Better |
| --- | --- | --- | --- | --- |
| Hard-check pass rate (target 1.000) | 0.000 | 0.897 | +0.897 | tuned |
| OOV rate (target <=0.02) | 0.157 | 0.007 | -0.150 | tuned |
| <=1 new word/sentence (target 1.000) | 0.000 | 1.000 | +1.000 | tuned |
| Recurrence satisfied (target 1.000) | 0.462 | 1.000 | +0.538 | tuned |
| Inferability (cloze; target 1.000) | 0.278 | 0.000 | -0.278 | base |

## LLM-as-judge rubric (0-2; first four are spec Appendix A)
| Metric | Base | Tuned | Delta | Better |
| --- | --- | --- | --- | --- |
| spec_adherence (target >=1.5) | 0.923 | 0.795 | -0.128 | base |
| robustness (target >=1.5) | 1.077 | 0.923 | -0.154 | base |
| task_quality (target >=1.5) | 1.436 | 0.103 | -1.333 | base |
| consistency (target >=1.5) | 1.795 | 1.077 | -0.718 | base |
| inferability (target >=1.5) | 1.410 | 0.769 | -0.641 | base |
| seductive_detail_control (target >=1.5) | 1.692 | 1.026 | -0.666 | base |
| coherence (target >=1.5) | 1.359 | 0.154 | -1.205 | base |
| interestingness (target >=1.5) | 1.308 | 0.000 | -1.308 | base |

## Win condition (spec)
Beats base on Spec adherence AND Robustness: **FAIL** (spec-adherence not up (judge_spec_adherence), robustness not up).

## Error analysis (tuned, held-out)
4/39 outputs failed a check. Most common:
- `oov`: 4
- `coverage`: 4

_Fill in: are the remaining failures a data problem (e.g. under-represented targets, themes that tempt off-vocab words)? What data change would fix them?_
---
_Legend: rates are fractions in [0,1]; judge scores in [0,2] (0 = fails, 1 = partial, 2 = fully). Hard-check pass rate, <=1-new-word, recurrence, coverage: **higher is better** (1.0 = every story passes). OOV rate: **lower is better** (0 = ideal; gate <= 0.02). OOV = out-of-vocabulary (a word not in the learner's known K or target T set); coverage = 1 - OOV rate._
