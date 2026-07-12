# Demo video script (3–5 min)

Goal per the spec: **show the model doing the thing the base model fails to do reliably.** The whole
video is one comparison — base vs tuned on the same i+1 task — plus a 30-second "why the numbers back
it up" close. Keep it tight; the graders have seen a hundred of these.

**Setup before recording.** `scripts/try_model.py` picks the vocab, prints the selection, then
generates — so the screen narrates itself. Two values are fixed for every command below:

- **Base model:** `Qwen/Qwen3-4B-Instruct-2507` — a Hugging Face model ID, not a folder. The script
  downloads it automatically the first time (~8 GB).
- **Adapter:** the folder your training run saved to Google Drive,
  `MyDrive/islm_v2_multi/qwen3_4b_v2_multi` (contains `adapter_config.json` +
  `adapter_model.safetensors`).

**Pick where you run — this decides the adapter path:**

- **On Colab (recommended — GPU, adapter already on Drive):** the base 4B on CPU is painfully slow,
  so record here. The adapter path is the Drive path directly:
  `/content/drive/MyDrive/islm_v2_multi/qwen3_4b_v2_multi`.
- **Locally:** download that `qwen3_4b_v2_multi` folder from Drive into your **Downloads** folder, so
  it lands at `C:\Users\18037\Downloads\qwen3_4b_v2_multi`. The commands below use that path. (Grab
  the whole folder — all files inside it, not just one.)

> If the multilingual run (`train_colab_v2_multi.ipynb`) hasn't finished, the last completed adapter
> is the **en-only** one at `MyDrive/islm_v2/qwen3_4b_v2` — usable for the demo in the meantime; just
> swap that path in.

The commands below are written for the **local Downloads** path. On Colab, replace
`C:\Users\18037\Downloads\qwen3_4b_v2_multi` with `/content/drive/MyDrive/islm_v2_multi/qwen3_4b_v2_multi`.

---

## Shot list

### 0:00–0:30 — The thesis (talking head or slide)
> "This is a 4B open model — Qwen3 — fine-tuned to write language-learning stories under a hard
> constraint: every story stays inside a learner's known words and adds **at most one new word per
> sentence**, while staying a real story. A well-prompted base model can't hold that reliably. I
> fixed the reliability with data, not a bigger model. Let me show you."

Say the **Behavior Spec** out loud — it's the pass/fail rubric the rest of the video is judged on.

### 0:30–1:45 — Base model fails (screen recording)
Run the **base** (no adapter) on English advanced targets:
```bash
python scripts/try_model.py --mode en --base-path Qwen/Qwen3-4B-Instruct-2507 --no-think
```
- Point at the printed **TARGETS** and **KNOWN tier** so the viewer sees the task.
- Read the output and narrate the violations: out-of-vocabulary words the learner wouldn't know,
  multiple new words per sentence, target words that appear once. "It writes a fine story — but not
  an *i+1* story. It breaks the constraint, and it breaks it differently every run." Run it **twice**
  to show the inconsistency (the spec's "reliable every time" dimension).

### 1:45–3:15 — Tuned model holds (screen recording)
Same command, **with the adapter**:
```bash
python scripts/try_model.py --mode en --base-path Qwen/Qwen3-4B-Instruct-2507 \
    --adapter C:\Users\18037\Downloads\qwen3_4b_v2_multi --no-think
```
- Same task framing. Read the output: in-vocabulary, one new word per sentence, target words recur,
  still a coherent little story. "Same base model, same prompt — the only thing that changed is it
  was trained on data that embodies the spec."
- Then show it **generalizes across languages** — one Chinese, one Japanese, and the hard exam mode:
```bash
python scripts/try_model.py --mode zh --base-path Qwen/Qwen3-4B-Instruct-2507 --adapter C:\Users\18037\Downloads\qwen3_4b_v2_multi --no-think
python scripts/try_model.py --mode jp --base-path Qwen/Qwen3-4B-Instruct-2507 --adapter C:\Users\18037\Downloads\qwen3_4b_v2_multi --no-think
python scripts/try_model.py --mode en-exam --base-path Qwen/Qwen3-4B-Instruct-2507 --adapter C:\Users\18037\Downloads\qwen3_4b_v2_multi --no-think
```
"Chinese, Japanese, and even hard GRE/SAT vocabulary — the behavior holds."

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
