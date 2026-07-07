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
