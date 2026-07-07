"""Build/refresh vocabulary tier files.

Frequency mode works for ~any language (wordfreq, Apache-2.0/MIT data), so the design is not
tied to English/Chinese/Japanese:

    python -m islm.vocab.build_lists --language en --from-frequency --overwrite

Writes data/vocab/<lang>/{baseline,advanced}.csv with columns word,tier,source. The curated
sample files shipped in the repo take precedence unless you pass --overwrite. Graded lists
(CEFR/HSK/JLPT) sourced from the web are documented in data/vocab/SOURCES.md.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .languages import get_language
from .wordlists import VOCAB_DIR


def _write_csv(path: Path, rows: list[tuple[str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "tier", "source"])
        writer.writerows(rows)


def build_from_frequency(language: str, overwrite: bool = False) -> None:
    """Baseline = top freq_baseline_n words; advanced = the next band up to freq_advanced_n."""
    from wordfreq import top_n_list  # lazy

    lang = get_language(language)
    words = top_n_list(language, lang.freq_advanced_n)
    out = VOCAB_DIR / language
    tiers = {
        "baseline.csv": [(w, "freq", "wordfreq") for w in words[: lang.freq_baseline_n]],
        "advanced.csv": [(w, "freq", "wordfreq") for w in words[lang.freq_baseline_n :]],
    }
    for name, rows in tiers.items():
        path = out / name
        if path.exists() and not overwrite:
            print(f"skip {path} (exists; pass --overwrite to replace)")
            continue
        _write_csv(path, rows)
        print(f"wrote {path} ({len(rows)} words)")


def main() -> None:
    p = argparse.ArgumentParser(description="Build vocabulary tier files.")
    p.add_argument("--language", required=True, help="Language code, e.g. en, zh, ja.")
    p.add_argument("--from-frequency", action="store_true", help="Generate from wordfreq bands.")
    p.add_argument("--overwrite", action="store_true", help="Replace existing files.")
    args = p.parse_args()

    if args.from_frequency:
        build_from_frequency(args.language, args.overwrite)
    else:
        p.error("specify a source, e.g. --from-frequency")


if __name__ == "__main__":
    main()
