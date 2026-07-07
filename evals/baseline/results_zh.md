# Eval: HuggingFaceTB/SmolLM2-135M-Instruct

Held-out scenarios: **8**.

## Behavioral checks (deterministic — the failures the spec forbids)
| Metric | Value |
| --- | --- |
| Hard-check pass rate | 0.000 |
| OOV rate | 0.856 |
| <=1 new word/sentence | 0.000 |
| Recurrence satisfied | 0.125 |

## Error analysis
8/8 outputs failed a check. Most common:
- `oov`: 8
- `coverage`: 8
- `one_new_word`: 8
- `recurrence`: 7
---
_Legend: rates are fractions in [0,1]; judge scores in [0,2] (0 = fails, 1 = partial, 2 = fully). Hard-check pass rate, <=1-new-word, recurrence, coverage: **higher is better** (1.0 = every story passes). OOV rate: **lower is better** (0 = ideal; gate <= 0.02). OOV = out-of-vocabulary (a word not in the learner's known K or target T set); coverage = 1 - OOV rate._
