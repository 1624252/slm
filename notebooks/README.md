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
| 3 | Regenerate `data/dataset_v1` from seeds (git-ignored but reproducible) |
| 4 | Smoke test — QLoRA loads Qwen3-4B in 4-bit, one step, one save |
| 5 | Full QLoRA fine-tune (rank 32 / α 64, lr 2e-4, cosine + merge) |
| 6 | Base-vs-tuned eval on golden + held-out — deterministic + judge + cloze, tracked |
| 7 | Zip and download the adapter + eval results |

Budget and unit discipline: see [`../docs/COLAB_PLAN.md`](../docs/COLAB_PLAN.md). Develop on L4,
reserve A100 for the single final run, disconnect the runtime as soon as a cell finishes.
