# Error analysis (Layer 3)

Error analysis is a human **systematically reviewing model outputs** to find where it goes wrong,
then naming those failure modes into a taxonomy. It's not a metric — it's the sensemaking that
*shapes* the metrics and coverage. It is the foundation the other layers build on (PDF Layer 3).

## The failure taxonomy (this project)

Every failure we've seen maps to one of these named modes. The first four are exactly the
deterministic validator failures (`src/islm/validators/`); the last two are found only by reading
the stories (no validator catches them yet — they're candidates for future checks).

| Mode | What it looks like | Caught by |
| --- | --- | --- |
| **oov-leak** | Uses a word outside `K ∪ T` (e.g. "blooming", "ambiance") | `coverage` (OOV rate) — automated |
| **pacing-break** | A sentence introduces ≥2 new words at once | `one_new_word` — automated |
| **under-recurrence** | A target word appears <3× (no spaced repetition) | `recurrence` — automated |
| **coverage-miss** | Overall coverage < 98% (too many unknown words) | `coverage` — automated |
| **degenerate-loop** | Repeats one sentence over and over to pad length | human review (not yet a check) |
| **off-theme / lesson-leak** | Ignores the theme, or announces "today we learn X" | human review (not yet a check) |

The automated modes are tallied for every run in the results JSON (`error_analysis` in
`src/islm/eval/report.py`) and printed under **"Error analysis"** in each `results_<lang>.md`.
That tally is the *seed* — it tells you which checks fire and how often. The human pass finds the
modes no check catches yet.

## The process (PDF Layer 3)

1. **Gather traces.** Use a real eval run's stored stories — `evals/<run>/results_<lang>.json`
   holds every generated story per scenario. (No model re-run needed; see Layer 4 replay.)
2. **Read them, journal open-endedly.** Don't score — note what's wrong in plain words. A human
   reads the traces; an LLM can't replace this judgment.
3. **Focus on the first failure in each trace.** Upstream errors cause downstream ones (e.g. an
   oov-leak in sentence 2 often drags coverage down for the whole story).
4. **Categorize into the taxonomy above.** Add a new named mode if something recurs and none fits.
5. **Iterate to saturation.** Stop when new traces stop revealing new modes.

Binary-first, scores-second: the deterministic pass/fail is the ground truth; the LLM-judge
rubric (Layer 5) is only trustworthy *after* it's calibrated against these human-found modes.

## Cadence

- **Automated tally:** every eval run (free, already in the report).
- **Human review:** at least weekly while actively iterating — review 20–50 outputs whenever a
  training change lands. A spreadsheet or the `results_<lang>.json` is enough; don't build eval
  infrastructure around an obvious bug — just fix it.

## What error analysis has already produced here

- The **prompt-truncation bug** (records were 5k–12k tokens; the story got truncated away) was
  found by reading Day-2/Day-3 traces and noticing the tuned model ignored the word list entirely
  — an *oov-leak* that no threshold tweak would fix. That became the compact-`KNOWN_WORDS` data
  fix (see `evals/RESULTS_LOG.md`).
- **degenerate-loop** was seen in early CPU runs (the model padding with a repeated sentence). Not
  yet a validator; flagged here so it isn't forgotten.
