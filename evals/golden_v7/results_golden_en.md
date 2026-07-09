# Eval results: base (HuggingFaceTB/SmolLM2-135M-Instruct) vs tuned (HuggingFaceTB/SmolLM2-135M-Instruct)

Held-out scenarios: **39**.

## Behavioral checks (deterministic — the failures the spec forbids)
| Metric | Base | Tuned | Delta | Better |
| --- | --- | --- | --- | --- |
| Hard-check pass rate (target 1.000) | 0.000 | 0.000 | +0.000 | base |
| OOV rate (target <=0.02) | 0.425 | 0.222 | -0.203 | tuned |
| <=1 new word/sentence (target 1.000) | 0.000 | 0.026 | +0.026 | tuned |
| Recurrence satisfied (target 1.000) | 0.205 | 0.103 | -0.103 | base |
| Inferability (cloze; target 1.000) | 0.179 | 0.051 | -0.128 | base |

## LLM-as-judge rubric (0-2; first four are spec Appendix A)
| Metric | Base | Tuned | Delta | Better |
| --- | --- | --- | --- | --- |
| spec_adherence (target >=1.5) | 0.000 | 0.103 | +0.103 | tuned |
| robustness (target >=1.5) | 0.000 | 0.077 | +0.077 | tuned |
| task_quality (target >=1.5) | 0.179 | 0.077 | -0.102 | base |
| consistency (target >=1.5) | 0.103 | 0.179 | +0.076 | tuned |
| inferability (target >=1.5) | 0.385 | 0.359 | -0.026 | base |
| seductive_detail_control (target >=1.5) | 0.231 | 0.256 | +0.025 | tuned |
| coherence (target >=1.5) | 0.205 | 0.205 | +0.000 | base |
| interestingness (target >=1.5) | 0.077 | 0.000 | -0.077 | base |

## Win condition (spec)
Beats base on Spec adherence AND Robustness: **PASS** (spec-adherence up (judge_spec_adherence), robustness up).

## Error analysis (tuned, held-out)
39/39 outputs failed a check. Most common:
- `one_new_word`: 38
- `oov`: 35
- `coverage`: 35
- `recurrence`: 35

_Fill in: are the remaining failures a data problem (e.g. under-represented targets, themes that tempt off-vocab words)? What data change would fix them?_
---
_Legend: rates are fractions in [0,1]; judge scores in [0,2] (0 = fails, 1 = partial, 2 = fully). Hard-check pass rate, <=1-new-word, recurrence, coverage: **higher is better** (1.0 = every story passes). OOV rate: **lower is better** (0 = ideal; gate <= 0.02). OOV = out-of-vocabulary (a word not in the learner's known K or target T set); coverage = 1 - OOV rate._
