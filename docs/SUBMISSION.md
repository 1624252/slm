# Submission package — status & checklist

Maps the spec's 5 required deliverables (`docs/spec.md` §"Final submission package") to where each
lives in this repo and what's left. Legend: ✅ done · ⚠️ needs a step · ❌ not started.

| # | Deliverable | Status | Where / how |
| --- | --- | --- | --- |
| 1 | **Dataset, published** | ⚠️ | In-repo: `data/dataset_v2/` (1610 stories, gz + `stats.json`) + `docs/DATA_CARD.md`. **To publish to HF:** verify HF account, then `python scripts/publish_hf.py dataset --repo <user>/islm-i-plus-1-stories`. |
| 2 | **Model on HF Hub + inference demo** | ⚠️ | Adapter is on Drive (`MyDrive/islm_v2_multi/qwen3_4b_v2_multi`). **To publish:** download it, then `python scripts/publish_hf.py model --repo <user>/... --adapter <dir>`. Demo = `scripts/try_model.py` (CLI); optional Gradio not built. |
| 3 | **Eval harness + results table (base vs tuned)** | ✅ | Harness: `src/islm/eval/`. Tables: `evals/LEADERBOARD.md` + `evals/RESULTS_LOG.md` (current, incl. v2-multi). Behavior metric = the judge rubric + deterministic checks. |
| 4 | **Brainlift** | ✅ | `docs/brainlift.md` — thesis + research tree + the new "Did data → behavior hold?" section with final evidence. |
| 5 | **3–5 min demo video** | ❌ | Script + shot list ready in `docs/DEMO_SCRIPT.md`. Recording is yours. |

## The behavior thesis (one-liner for graders)

Same 4B base (`Qwen/Qwen3-4B-Instruct-2507`), prompt vs. fine-tune: a well-prompted base model can't
reliably write i+1 stories (stay in known vocab, ≤1 new word/sentence, recur targets, stay a story);
the fine-tune does, and every quality problem was traced to and fixed in the **data**. Numbers on the
board in `evals/LEADERBOARD.md`.

## Remaining steps, in order

1. **Retrain (optional but recommended)** on the finalized 1610-story set (`train_colab_v2_multi.ipynb`)
   — the widened-ja-palette data is meant to fix ja prose flatness. Pull results, log, done.
2. **Verify the HF account** (the earlier `create_repo` 500s were an account block — check
   <https://huggingface.co/settings/account>, confirm email).
3. **Publish dataset:** `python scripts/publish_hf.py dataset --repo <user>/islm-i-plus-1-stories`.
4. **Download the adapter** from Drive, then **publish model:**
   `python scripts/publish_hf.py model --repo <user>/islm-i-plus-1-qwen3-4b --adapter <dir>`.
5. **Record the demo** following `docs/DEMO_SCRIPT.md`.
6. (Optional) A hosted Gradio demo for deliverable #2's "running inference demo" — `try_model.py`
   covers the behavior, but a link is stronger.

## Already solid (no action)

- Behavior Spec — `docs/EVALUATION.md` §"The Behavior Spec"; falsifiable 0/1/2 judge anchors in
  `src/islm/llm/prompts.py`.
- Base-vs-tuned across 3 languages, golden + held-out + adversarial — `evals/`.
- Dataset provenance, filtering, reproduction — `docs/DATA_CARD.md`, `docs/dataset-and-eval.md`.
- Error analysis — `docs/ERROR_ANALYSIS.md` + the per-run "Read" sections in `evals/RESULTS_LOG.md`.
