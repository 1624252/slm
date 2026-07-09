# Eval results: base (HuggingFaceTB/SmolLM2-135M-Instruct) vs tuned (HuggingFaceTB/SmolLM2-135M-Instruct)

Held-out scenarios: **39**.

## Behavioral checks (deterministic — the failures the spec forbids)
| Metric | Base | Tuned | Delta | Better |
| --- | --- | --- | --- | --- |
| Hard-check pass rate (target 1.000) | 0.000 | 0.103 | +0.103 | tuned |
| OOV rate (target <=0.02) | 0.425 | 0.133 | -0.292 | tuned |
| <=1 new word/sentence (target 1.000) | 0.000 | 0.487 | +0.487 | tuned |
| Recurrence satisfied (target 1.000) | 0.205 | 0.385 | +0.179 | tuned |
| Inferability (cloze; target 1.000) | 0.179 | 0.000 | -0.179 | base |

## LLM-as-judge rubric (0-2; first four are spec Appendix A)
| Metric | Base | Tuned | Delta | Better |
| --- | --- | --- | --- | --- |
| spec_adherence (target >=1.5) | 0.000 | 0.590 | +0.590 | tuned |
| robustness (target >=1.5) | 0.000 | 0.513 | +0.513 | tuned |
| task_quality (target >=1.5) | 0.154 | 0.000 | -0.154 | base |
| consistency (target >=1.5) | 0.103 | 1.000 | +0.897 | tuned |
| inferability (target >=1.5) | 0.359 | 0.821 | +0.462 | tuned |
| seductive_detail_control (target >=1.5) | 0.256 | 0.513 | +0.257 | tuned |
| coherence (target >=1.5) | 0.231 | 0.154 | -0.077 | base |
| interestingness (target >=1.5) | 0.051 | 0.000 | -0.051 | base |

## Win condition (spec)
Beats base on Spec adherence AND Robustness: **PASS** (spec-adherence up (judge_spec_adherence), robustness up).

## Error analysis (tuned, held-out)
35/39 outputs failed a check. Most common:
- `oov`: 30
- `coverage`: 30
- `recurrence`: 24
- `one_new_word`: 20

_Fill in: are the remaining failures a data problem (e.g. under-represented targets, themes that tempt off-vocab words)? What data change would fix them?_
---
_Legend: rates are fractions in [0,1]; judge scores in [0,2] (0 = fails, 1 = partial, 2 = fully). Hard-check pass rate, <=1-new-word, recurrence, coverage: **higher is better** (1.0 = every story passes). OOV rate: **lower is better** (0 = ideal; gate <= 0.02). OOV = out-of-vocabulary (a word not in the learner's known K or target T set); coverage = 1 - OOV rate._
