# Behavioral coverage matrix (Layer 2)

Golden set: **55** tagged cases. Empty cells (`-`) show where to add tests next. Tags come from `golden.py`; regenerate with `python -m islm.eval.coverage_matrix`.

## Language x difficulty

|                | straightforward | ambiguous       | edge            |
|----------------|-----------------|-----------------|-----------------|
| en             | 35              | 3               | 1               |
| zh             | 8               | -               | -               |
| ja             | 8               | -               | -               |

## Category x difficulty

|                | straightforward | ambiguous       | edge            |
|----------------|-----------------|-----------------|-----------------|
| cefr           | 12              | 1               | -               |
| core           | 28              | 2               | -               |
| exam           | 11              | -               | -               |
| mixed          | -               | -               | 1               |

## Target-tier coverage

| Tier | Cases |
| --- | --- |
| ACT | 1 |
| B2 | 10 |
| C1 | 5 |
| C2 | 1 |
| GRE | 3 |
| SAT | 8 |
| core | 32 |

_difficulty: straightforward = 1 target, ambiguous = 2, edge = 3+ new words to pace._
