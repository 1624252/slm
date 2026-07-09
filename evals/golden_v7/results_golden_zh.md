# Eval results: base (HuggingFaceTB/SmolLM2-135M-Instruct) vs tuned (HuggingFaceTB/SmolLM2-135M-Instruct)

Held-out scenarios: **8**.

## Behavioral checks (deterministic — the failures the spec forbids)
| Metric | Base | Tuned | Delta | Better |
| --- | --- | --- | --- | --- |
| Hard-check pass rate (target 1.000) | 0.000 | 0.125 | +0.125 | tuned |
| OOV rate (target <=0.02) | 0.881 | 0.359 | -0.522 | tuned |
| <=1 new word/sentence (target 1.000) | 0.000 | 0.500 | +0.500 | tuned |
| Recurrence satisfied (target 1.000) | 0.000 | 0.375 | +0.375 | tuned |
| Inferability (cloze; target 1.000) | 0.000 | 0.125 | +0.125 | tuned |

## LLM-as-judge rubric (0-2; first four are spec Appendix A)
| Metric | Base | Tuned | Delta | Better |
| --- | --- | --- | --- | --- |
| spec_adherence (target >=1.5) | 0.000 | 0.125 | +0.125 | tuned |
| robustness (target >=1.5) | 0.000 | 0.000 | +0.000 | base |
| task_quality (target >=1.5) | 0.500 | 0.000 | -0.500 | base |
| consistency (target >=1.5) | 0.000 | 0.000 | +0.000 | base |
| inferability (target >=1.5) | 0.000 | 0.125 | +0.125 | tuned |
| seductive_detail_control (target >=1.5) | 0.000 | 0.000 | +0.000 | base |
| coherence (target >=1.5) | 0.500 | 0.000 | -0.500 | base |
| interestingness (target >=1.5) | 0.125 | 0.000 | -0.125 | base |

## Win condition (spec)
Beats base on Spec adherence AND Robustness: **FAIL** (spec-adherence up (judge_spec_adherence), robustness not up).

## Error analysis (tuned, held-out)
7/8 outputs failed a check. Most common:
- `oov`: 5
- `coverage`: 5
- `recurrence`: 5
- `one_new_word`: 4

_Fill in: are the remaining failures a data problem (e.g. under-represented targets, themes that tempt off-vocab words)? What data change would fix them?_
---
_Legend: rates are fractions in [0,1]; judge scores in [0,2] (0 = fails, 1 = partial, 2 = fully). Hard-check pass rate, <=1-new-word, recurrence, coverage: **higher is better** (1.0 = every story passes). OOV rate: **lower is better** (0 = ideal; gate <= 0.02). OOV = out-of-vocabulary (a word not in the learner's known K or target T set); coverage = 1 - OOV rate._
