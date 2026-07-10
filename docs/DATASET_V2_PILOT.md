# Dataset v2 — teacher-regen quality pilot

## Why

The `dataset_v1` stories are built by `islm.datagen.synth` from meaning-blind templates. They pass
the deterministic validators but read as nonsense — `"To be forlorn is good."` (forlorn = miserable),
`"The scroll is a good friend."`, `"The meadow is dark and bright. The meadow is dark."`. The first
GPU QLoRA run (see `evals/RESULTS_LOG.md`, 2026-07-10) confirmed the cost: the tuned model **solved
the mechanical i+1 constraints** (en golden hard-pass 0.00 → 0.90) but **regressed on judge quality**
(coherence, task_quality, interestingness collapsed) — it learned to game the checks with dull text,
because that is exactly what the training data looks like.

The fix is at the source: better data, not more training.

## Approach — `islm.datagen.teacher`

Ask a real LLM teacher to *write* each i+1 story (coherent prose), run the validator-guided rewrite
loop until it hard-passes, then apply a **judge gate** (coherence = 2, task_quality = 2,
interestingness ≥ 1) so flat/gamed stories are dropped. Records use the same compact-KNOWN
convention and schema as `dataset_v1`, so training and eval read them unchanged.

This is a **pilot** (~2–3k stories) to prove regenerated data lifts the judge scores before
spending on full-scale generation.

## Run the pilot (needs `OPENAI_API_KEY` + judge model in `.env`)

```bash
# ~1k per language; the judge gate drops low-quality drafts (keep_rate < 1).
PYTHONPATH=src python -m islm.datagen.teacher --n 1000 --language en --out data/generated/teacher_en
PYTHONPATH=src python -m islm.datagen.teacher --n 1000 --language zh --out data/generated/teacher_zh
PYTHONPATH=src python -m islm.datagen.teacher --n 1000 --language ja --out data/generated/teacher_ja

# Corpus-level dedup / defense-in-depth (reuses the existing second pass):
PYTHONPATH=src python -m islm.datagen.curate --in data/generated/teacher_en --out data/curated/teacher_en
# ... repeat for zh, ja ...
```

Offline smoke (no key, deterministic MockLLM — proves the plumbing only):

```bash
PYTHONPATH=src python -m islm.datagen.teacher --n 5 --language en --mock --out /tmp/teacher_en
```

Flags: `--teacher-model` / `--judge-model` override the env defaults; `--no-judge` skips the quality
gate (deterministic-only); `--max-attempts` caps API spend (default `4 × n`).

## Validate it actually improved

1. Assemble the curated pilot into a `data/dataset_v2/` (same layout as v1).
2. Eyeball 20 stories — they should read as real, coherent mini-stories, not filler.
3. Train a small adapter on the pilot and eval (golden + held-out). The signal to look for vs the
   v1 GPU run: **judge coherence / task_quality / interestingness go UP** while the deterministic
   constraints stay high. If so, scale up generation; if not, tighten the prompt/gate first.

## Cost control

- Each kept story ≈ 1 generation + up to `max_rewrites` fixes + 1 judge call. Budget by
  `keep_rate` (printed in `teacher_stats.json`).
- Start with one language (en) to sanity-check `keep_rate` and prose quality before spending on all
  three.
