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
- v6 (epochs 3): DONE — NEGATIVE result, worse than v5 on golden (zh/ja OOV regressed). Committed.
  Confirms coherence/interestingness collapse is model-capacity, not over-fitting.
- Teacher throughput measured: **~28s/scenario, 40% keep** (unbuffered `python -u`). Low CPU is
  normal (API-bound) — don't mistake it for a stall.
- Data plan: en=80 (→~32 kept) generating; then zh/ja if budget allows. Combine with 22-record
  seed → train v7 on the larger corpus.
- Budget ~02:05, ~5.9h left. Reserve: v7 train ~2h + eval ~40m + summary 20m. Data-gen window ~2h.
- Teacher yields: en 48/80 (60%), zh 30/40 (75%), ja 4/40 (10% — CJK is hard for the teacher too).
- **Teacher KNOWN_WORDS bug found + fixed**: teacher recs were ~5.2k tokens (full baseline); new
  `islm.datagen.compact` shrinks to ~830 tok. MUST compact teacher data before training.
- **v7 corpus** = compact teacher(en+zh+ja) + seed = **87 train** (en50/zh30/ja7), 4× the seed.
- **v7 training** (~03:15): 3 epochs capped at **--max-steps 200** (budget), seq 1024, else = v5
  (r32/a64/lr2e-4). DATA is the only real variable. ~3h → done ~06:15, leaves eval+summary time.
- v7 recapped to --max-steps 90 (seq-1024 was ~96s/step; 200 would overrun budget). Restarted ~04:07, done ~06:30. Data is the only variable vs v5.
