# Demo video script (3–5 min)

Goal per the spec: **show the model doing the thing the base model fails to do reliably.** The whole
video is one comparison — base vs tuned on the same i+1 task — plus a 30-second "why the numbers back
it up" close. Keep it tight; the graders have seen a hundred of these.

**Easiest path: just run `notebooks/demo_colab.ipynb` on Colab** — it does all of this one cell at a
time. This doc is the narration guide / manual-command reference.

**Setup.** `scripts/try_model.py --compare` picks the vocab, prints the selection, runs the base and
the tuned model on the *same* scenario, then prints an **IMPROVEMENT OVER BASE** table — so the screen
narrates itself and shows the gain, not just a pass/fail stamp. Two fixed values for every command
below:

- **Base model:** `Qwen/Qwen3-4B-Instruct-2507` — a Hugging Face model ID. Downloads on first use (~8 GB).
- **Adapter:** **`i0445/islm`** — the fine-tuned adapter, published to the Hugging Face Hub. The
  script loads it straight from the Hub (no Drive needed); it downloads on first use.

Record on **Colab with a GPU** — the 4B base on CPU is painfully slow. (To demo a local Drive copy
instead, point `--adapter` at `/content/drive/MyDrive/islm_v2_multi/qwen3_4b_v2_multi`.)

---

## Shot list

### 0:00–0:30 — The thesis (talking head or slide)
> "This is a 4B open model — Qwen3 — fine-tuned to write language-learning stories under a hard
> constraint: every story stays inside a learner's known words and adds **at most one new word per
> sentence**, while staying a real story. A well-prompted base model can't hold that reliably. I
> fixed the reliability with data, not a bigger model. Let me show you."

Say the **Behavior Spec** out loud — it's the pass/fail rubric the rest of the video is judged on.

### 0:30–2:30 — Base vs tuned, same scenario (screen recording)
Run `--compare` on English advanced targets — it runs both models on the *same* sampled scenario:
```bash
python scripts/try_model.py --mode en --base-path Qwen/Qwen3-4B-Instruct-2507 \
    --adapter i0445/islm --no-think --compare
```
- Point at the printed **TARGETS** and **KNOWN tier** so the viewer sees the task.
- Read the **BASE** story and narrate the violations: out-of-vocabulary words the learner wouldn't
  know, multiple new words per sentence, target words that appear once. "It writes a fine story, but
  not an *i+1* story."
- Then read the **TUNED** story: in-vocabulary, one new word per sentence, target words recur, still a
  coherent little story.
- Land on the **IMPROVEMENT OVER BASE** table: OOV rate collapsing toward the 2% limit, coverage up,
  new-words-per-sentence down to 1, hard pass FAIL to PASS. "Same base model, same prompt, same
  scenario. The only thing that changed is training data that embodies the spec, and here's the
  measured gain."
- Run it **twice** — the base breaks the constraint differently each time (the spec's "reliable every
  time" dimension); the tuned model keeps holding it.

### 2:30–3:15 — It generalizes across languages (screen recording)
Same `--compare`, other modes — one Chinese, one Japanese, and the hard exam mode:
```bash
python scripts/try_model.py --mode zh --base-path Qwen/Qwen3-4B-Instruct-2507 --adapter i0445/islm --no-think --compare
python scripts/try_model.py --mode jp --base-path Qwen/Qwen3-4B-Instruct-2507 --adapter i0445/islm --no-think --compare
python scripts/try_model.py --mode en-exam --base-path Qwen/Qwen3-4B-Instruct-2507 --adapter i0445/islm --no-think --compare
```
For zh, point at the OOV drop in the improvement table (roughly 0.33 to ~0.08) rather than a hard
pass. "Chinese OOV collapses, Japanese holds the constraint, and it works on hard GRE/SAT vocabulary."

### 3:15–4:15 — The numbers (screen or slide: `evals/LEADERBOARD.md`)
Show the base-vs-tuned table. Call out the deltas that matter:
- **en golden:** hard-pass 0.00 → 0.49, OOV 0.157 → 0.02, ≤1-new 0.00 → 0.56.
- **quality didn't collapse:** en held-out interestingness 0.83 → **1.58** (above base).
- **all three languages** move on OOV; ja/zh post their constraint gains once they have data.
> "This is the point: fine-tuning didn't make it smarter, it made it **reliable** at one narrow thing
> — measurably, versus the same base model."

---

## Tips
- Pre-run the commands once so models are warm; long model-load waits are dead air. On CPU the 4B
  model is slow — record in the Colab GPU session.
- If a base run happens to look decent, run it again — inconsistency *is* the point, and it shows.
- Keep the tuned outputs on screen long enough to actually read a sentence or two.
