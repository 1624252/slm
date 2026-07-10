# Notebooks

## `train_colab.ipynb` — QLoRA fine-tuning on Colab GPU

End-to-end GPU training + eval for the i+1 story model. Open it in Colab:

1. Go to <https://colab.research.google.com>, **File → Open notebook → GitHub**, and paste this
   repo's URL (or **Upload** the `.ipynb`).
2. **Runtime → Change runtime type → L4 GPU.**
3. Add your keys in the **Secrets** panel (key icon, left sidebar) — see Step 2 in the notebook for
   the exact names. Never paste keys into cells.
4. Run the cells top to bottom.

What it does (each step is one cell group):

| Step | Action |
| --- | --- |
| 0–1 | Confirm GPU, clone the repo, install `.[train]` + `bitsandbytes` |
| 2 | Load API keys from Colab Secrets into the environment |
| 2.5 | **Mount Google Drive** — durable auto-save for the adapter, checkpoints, and results |
| 3 | Decompress the shipped `data/dataset_v1` (already cloned; no regeneration) |
| 4 | Smoke test — QLoRA loads Qwen3-4B in 4-bit, one step, one save |
| 5 | Full QLoRA fine-tune (rank 32 / α 64, lr 2e-4, cosine), checkpointing to Drive |
| 6 | Base-vs-tuned eval on golden + held-out — deterministic + judge + cloze, tracked |
| 6b | Push results to LangSmith (no-ops without the key) |
| 7 | (Optional) zip + download a local copy — everything is already on Drive |

### Crash-safe by default

Colab reclaims idle runtimes and wipes their ephemeral disk — that's how a finished run can vanish.
Step 2.5 mounts Drive and points the training output, its **periodic checkpoints** (`--save-steps
200`), and the eval results at `MyDrive/islm/`, so they persist **as they're written**. If the
runtime dies mid-train, just **re-run Step 5** — `islm.train.sft` auto-resumes from the last
checkpoint on Drive. Set `USE_DRIVE = False` (Step 2.5) to fall back to the old ephemeral + manual
download behavior.

Budget and unit discipline: see [`../docs/COLAB_PLAN.md`](../docs/COLAB_PLAN.md). Develop on L4,
reserve A100 for the single final run, disconnect the runtime as soon as a cell finishes.
