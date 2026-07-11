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

## Pilot status (2026-07-11)

**`data/dataset_v2/` = EXAM-target English set, humanized (current).** Regenerated with
`--targets exam` so the target words are the GRE/SAT/ACT list (`data/vocab/en/exam.csv`), which is
what the graded golden set's harder tier tests. **600 stories** (480/60/60 train/val/test), 593
unique exam targets, 2.34 targets/story, **0 story leakage**, all spec-passing. Quality held even on
hard vocab: **coherence = 2 and task_quality = 2 for 100%** of records, inferability = 2 for ~90%.
Keep-rate 0.89; curate dedup kept 100%.

**Humanizer pass:** the teacher leaned on the em-dash gloss (`stolid mouse — nothing could make him
run`) in 570/600 stories — the top AI tell per the humanizer skill (§14). A follow-up pass removed
every em/en dash (splitting each into two short sentences, which also suits i+1 pacing) **without
changing any words, so every target word is preserved by construction**, then re-validated all 600
against the full spec and re-checked the judge gate; **0 dropped, 0 residual dashes**. Sample after:
*"It ran in a **tangential** way, not straight, but out to the side."*; *"she has **fortitude**. She
stays in the tree and does not run away."*

(The earlier graded-target pilot — 594 concrete-word stories — was overwritten here per request; it
still exists at `data/generated/teacher_en/` if needed. Exam generation source:
`data/generated/teacher_en_exam/`.)

**zh/ja added (2026-07-11).** `data/dataset_v2/` is now **multilingual**: en 600 + **zh 202** +
**ja 185** = 987 stories. zh/ja have no exam vocab, so their targets are the graded concrete-noun
pools (`synth.TARGET_POOLS`: lighthouse, rainbow, castle, treasure, robot, …). Unlike the API
teacher used for en, these were **authored directly (no OpenAI API — zero contention with a running
Colab eval)** and validated through the same deterministic pipeline via `scripts/author_cjk.py`:
compact-known scoping (coverage passes by construction) + the hard checks (OOV, ≤1-new-word/sentence,
recurrence ≥3×). Every kept story hard-passes; a 40-record random re-validation of the merged set is
40/40. zh 41 unique targets, ja 42; 0 story leakage across splits. (CJK has no em-dash tell, so no
humanizer pass is needed there.) The eval already runs all three languages, so the multilingual v2
lets the A/B measure zh/ja quality too — not just en.

## Validate it actually improved — the comparison train (A/B)

Goal: same recipe as the v1 GPU run, only the **data** changes, so any judge-quality lift is
attributable to the dataset. Train **from base** (a clean A/B — do *not* continue the v1 adapter, or
the comparison is confounded).

On Colab (or any GPU), the only change vs the standard notebook flow is the `--data` path and a fresh
output dir:

```bash
python -m islm.train.sft --data data/dataset_v2 --base Qwen/Qwen3-4B-Instruct-2507 --qlora \
    --max-steps 1500 --lr 2e-4 --lora-r 32 --lora-alpha 64 \
    --max-seq-len 1024 --grad-accum 8 --out outputs/qwen3_4b_v2

python -m islm.eval.run --golden --base-path Qwen/Qwen3-4B-Instruct-2507 \
    --tuned-path Qwen/Qwen3-4B-Instruct-2507 --tuned-adapter outputs/qwen3_4b_v2 \
    --no-think --max-new-tokens 320 --track --run-label v2-en-golden \
    --dataset data/dataset_v2 --out evals/v2_golden
# ... and the held-out eval, same as the notebook's Step 6 ...
```

**The signal that decides go/no-go** — compare tuned judge scores against the v1 run #2 (see
`evals/RESULTS_LOG.md`, 2026-07-11):

| | v1 iter #2 (en golden, tuned) | v2 target |
| --- | --- | --- |
| hard-pass | 0.97 | stay ≥ ~0.9 |
| coherence | **0.23** | **↑ (toward 1.5+)** |
| task_quality | **0.26** | **↑** |
| interestingness | **0.03** | **↑** |

If coherence/interestingness rise materially while the constraints hold, the data hypothesis is
confirmed → scale generation (top up en, add zh/ja, grow toward the v1 size). If not, tighten the
teacher prompt / judge gate before spending more.

*(dataset_v2 is en-only for now; the eval runs all languages, so expect zh/ja tuned rows to look
like the base until their pilots exist. Judge the experiment on the **en** rows.)*

## Cost control

- Each kept story ≈ 1 generation + up to `max_rewrites` fixes + 1 judge call. Budget by
  `keep_rate` (printed in `teacher_stats.json`).
- Start with one language (en) to sanity-check `keep_rate` and prose quality before spending on all
  three.
