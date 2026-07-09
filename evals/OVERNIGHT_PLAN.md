# Overnight autonomous training plan (2026-07-09)

**Budget:** started ~00:21, HARD STOP **07:59 AM**. Leave ≥20 min at the end to document + commit.
**Goal:** the model must **improve over v5** (current best) by morning. Improvement =
lower golden-set OOV AND no collapse of coherence/interestingness (the regression every run hit).

## Current best to beat — v5 (golden set, base→tuned)
- en OOV 0.425→0.133, ja OOV 1.00→0.117, zh OOV 0.881→0.250; en/ja hard-pass up.
- BUT coherence/interestingness dropped to ~0. That is the thing to fix, not just OOV.

## Strategy (highest-leverage first)
The teacher endpoint WORKS (`.env` TEACHER_MODEL, live Claude). Re-tuning hyperparameters on 22
examples has low ceiling (v3→v4→v5 showed it). The real lever is **more + more-varied data**.

1. **v6** (running): epochs 3 vs v5's 5 — last pure-HP probe. Eval, log, keep if better.
2. **Scale data with the teacher** — generate a larger curated corpus (target a few hundred
   spec-passing records across en/zh/ja) via `islm.datagen.pipeline`, curated + validated. This is
   the PRD's core deliverable. Held-out/golden stay untouched (no leakage).
3. **v7+**: train on the bigger corpus. Iterate one variable at a time; keep the best on golden.
4. Always: golden gate (pytest) before/after; eval all 3 surfaces judged; RESULTS_LOG + commit
   EACH run so progress is durable even if interrupted.

## Loop discipline (from the train-islm skill)
- One variable per iteration. Log every attempt incl. failures.
- Commit after every completed run (tree clean is the checkpoint).
- CPU ~90s/step. Budget runs so the LAST one finishes + documents before 07:59.
- If a run is mid-flight near the deadline, let the current one finish, then STOP and write the
  summary — do not start a run that can't finish + commit in time.

## Progress log (update as we go)
- v6 (epochs 3): training…
