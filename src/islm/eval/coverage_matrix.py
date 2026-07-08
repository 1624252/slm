"""Behavioral coverage matrix (Layer 2) — where are we tested, and where are the gaps?

Reads the golden set's coverage tags (category / difficulty / language / tier, added in
`golden.py`) and renders a matrix. The tags don't change how anything runs — they change what the
results tell you: **empty cells show where to write the next test** (PDF Layer 2). No model calls.

    python -m islm.eval.coverage_matrix                       # writes evals/golden/coverage.md
    python -m islm.eval.coverage_matrix --print               # just print to stdout
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from ..config import EVALS_DIR

DIFFICULTIES = ["straightforward", "ambiguous", "edge"]
LANGUAGES = ["en", "zh", "ja"]


def load_golden(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _grid(records: list[dict], row_key, rows: list[str], cols: list[str], col_key) -> list[str]:
    """Render a counts table: rows x cols, with '-' for empty cells (gaps to fill)."""
    counts: Counter = Counter()
    for r in records:
        counts[(row_key(r), col_key(r))] += 1
    width = max([len(c) for c in cols] + [7])
    header = "| " + " " * 14 + " | " + " | ".join(c.ljust(width) for c in cols) + " |"
    sep = "|" + "-" * 16 + "|" + "|".join("-" * (width + 2) for _ in cols) + "|"
    lines = [header, sep]
    for row in rows:
        cells = []
        for col in cols:
            n = counts[(row, col)]
            cells.append((str(n) if n else "-").ljust(width))
        lines.append("| " + row.ljust(14) + " | " + " | ".join(cells) + " |")
    return lines


def render(records: list[dict]) -> str:
    """Two views: language x difficulty, and category x difficulty. Empty cells = coverage gaps."""
    total = len(records)
    tiers = Counter(t for r in records for t in r["metadata"]["target_tier"])
    out = [
        "# Behavioral coverage matrix (Layer 2)",
        "",
        f"Golden set: **{total}** tagged cases. Empty cells (`-`) show where to add tests next. "
        "Tags come from `golden.py`; regenerate with `python -m islm.eval.coverage_matrix`.",
        "",
        "## Language x difficulty",
        "",
        *_grid(
            records,
            row_key=lambda r: r["language"],
            rows=LANGUAGES,
            cols=DIFFICULTIES,
            col_key=lambda r: r["metadata"]["difficulty"],
        ),
        "",
        "## Category x difficulty",
        "",
        *_grid(
            records,
            row_key=lambda r: r["metadata"]["category"],
            rows=sorted({r["metadata"]["category"] for r in records}),
            cols=DIFFICULTIES,
            col_key=lambda r: r["metadata"]["difficulty"],
        ),
        "",
        "## Target-tier coverage",
        "",
        "| Tier | Cases |",
        "| --- | --- |",
        *[f"| {tier} | {n} |" for tier, n in sorted(tiers.items())],
        "",
        "_difficulty: straightforward = 1 target, ambiguous = 2, edge = 3+ new words to pace._",
    ]
    return "\n".join(out) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="Render the golden-set behavioral coverage matrix.")
    p.add_argument("--golden", type=Path, default=EVALS_DIR / "golden" / "golden.jsonl")
    p.add_argument("--out", type=Path, default=EVALS_DIR / "golden" / "coverage.md")
    p.add_argument("--print", action="store_true", dest="to_stdout", help="Print instead of write.")
    args = p.parse_args()

    md = render(load_golden(args.golden))
    if args.to_stdout:
        print(md)
    else:
        args.out.write_text(md, encoding="utf-8")
        print(f"coverage matrix -> {args.out}")


if __name__ == "__main__":
    main()
