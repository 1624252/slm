"""Download real graded word lists into ``data/vocab/<lang>/{baseline,advanced}.full.csv``.

Loaders prefer these ``*.full.csv`` files over the small committed curated samples. The full
files are git-ignored (fetched on demand) — see ``data/vocab/SOURCES.md`` for exact licenses.

Sources (all verified open + machine-readable):
- en: CEFR-J Vocabulary Profile 1.5 + Octanove C1/C2  (openlanguageprofiles/olp-en-cefrj)
      plus the committed GRE/SAT/ACT list (data/vocab/en/exam.csv) for exam-level hard words
- zh: HSK 3.0                                          (ivankra/hsk30, MIT)
- ja: JLPT N5-N1                                       (evanclan/OpenJLPT, CC-BY-SA-4.0)

    python -m islm.vocab.download --language all

Frequency baselines (hermitdave/FrequencyWords, CC-BY-SA-4.0) for any language are available
separately via ``islm.vocab.build_lists --from-frequency``.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import urllib.request

from .wordlists import VOCAB_DIR

_UA = {"User-Agent": "islm-vocab-downloader"}
_EN_RE = re.compile(r"^[a-z][a-z'\u2019-]*$")  # single lower-case English word
# All-Han (Chinese) and all-Japanese-script words; drops ellipsis/ASCII/mojibake noise.
_ZH_RE = re.compile(r"^[\u4e00-\u9fff\u3400-\u4dbf]+$")
_JA_RE = re.compile(r"^[\u3040-\u30ff\u4e00-\u9fff\u3400-\u4dbf\u30fc]+$")

_CEFRJ = "https://raw.githubusercontent.com/openlanguageprofiles/olp-en-cefrj/master/cefrj-vocabulary-profile-1.5.csv"
_OCTANOVE = "https://raw.githubusercontent.com/openlanguageprofiles/olp-en-cefrj/master/octanove-vocabulary-profile-c1c2-1.0.csv"
_HSK = "https://raw.githubusercontent.com/ivankra/hsk30/master/hsk30.csv"
_JLPT = "https://raw.githubusercontent.com/evanclan/OpenJLPT/main/data/csv/vocab-{lvl}.csv"

# tier -> (baseline?, advanced?)
_EN_BASELINE = {"A1", "A2"}
_ZH_BASELINE = {"1", "2", "3"}
_JA_BASELINE = {"N5", "N4"}


def _get(url: str) -> str:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 - trusted raw URLs
        return resp.read().decode("utf-8")


def _rows(text: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(text)))


def _write(language: str, tier_words: dict[str, list[tuple[str, str, str]]]) -> dict[str, int]:
    """Write baseline/advanced full CSVs, keeping the tiers disjoint (baseline wins)."""
    baseline = _dedupe(tier_words["baseline"])
    advanced = _dedupe(tier_words["advanced"])
    advanced_rows = [row for word, row in advanced.items() if word not in baseline]
    out = VOCAB_DIR / language
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("baseline", list(baseline.values())), ("advanced", advanced_rows)):
        with open(out / f"{name}.full.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["word", "tier", "source"])
            w.writerows(rows)
    return {"baseline": len(baseline), "advanced": len(advanced_rows)}


def _dedupe(rows: list[tuple[str, str, str]]) -> dict[str, tuple[str, str, str]]:
    out: dict[str, tuple[str, str, str]] = {}
    for word, tier, source in rows:
        if word and word not in out:
            out[word] = (word, tier, source)
    return out


def load_exam_rows() -> list[tuple[str, str, str]]:
    """Committed GRE/SAT/ACT words from `data/vocab/en/exam.csv` (empty if the file is absent).

    These are exam-level hard words the CEFR-J/Octanove lists miss. Appended AFTER the graded
    rows so `_dedupe` keeps a word's CEFR tier when it already has one (see `download_en`).
    """
    path = VOCAB_DIR / "en" / "exam.csv"
    if not path.exists():
        return []
    rows: list[tuple[str, str, str]] = []
    for row in _rows(path.read_text(encoding="utf-8")):
        word = (row.get("word") or "").strip().lower()
        if _EN_RE.match(word):
            rows.append((word, (row.get("tier") or "EXAM").strip(), row.get("source") or "exam"))
    return rows


def download_en() -> dict[str, int]:
    tiers: dict[str, list] = {"baseline": [], "advanced": []}
    for url, src in ((_CEFRJ, "cefr-j-1.5"), (_OCTANOVE, "octanove-c1c2-1.0")):
        for row in _rows(_get(url)):
            level = (row.get("CEFR") or "").strip().upper()
            for variant in (row.get("headword") or "").split("/"):
                word = variant.strip().lower()
                if not _EN_RE.match(word):
                    continue
                bucket = "baseline" if level in _EN_BASELINE else "advanced"
                tiers[bucket].append((word, level, src))
    # Exam words extend the advanced tier; graded rows come first so CEFR tiers win on dedup.
    tiers["advanced"].extend(load_exam_rows())
    return _write("en", tiers)


def download_zh() -> dict[str, int]:
    tiers: dict[str, list] = {"baseline": [], "advanced": []}
    for row in _rows(_get(_HSK)):
        word = (row.get("Simplified") or "").strip()
        level = (row.get("Level") or "").strip()
        if not _ZH_RE.match(word):
            continue
        bucket = "baseline" if level in _ZH_BASELINE else "advanced"
        tiers[bucket].append((word, f"HSK{level}", "hsk30"))
    return _write("zh", tiers)


def download_ja() -> dict[str, int]:
    tiers: dict[str, list] = {"baseline": [], "advanced": []}
    for lvl in ("n5", "n4", "n3", "n2", "n1"):
        for row in _rows(_get(_JLPT.format(lvl=lvl))):
            word = (row.get("word") or "").strip()
            level = (row.get("level") or lvl.upper()).strip().upper()
            if not _JA_RE.match(word):
                continue
            bucket = "baseline" if level in _JA_BASELINE else "advanced"
            tiers[bucket].append((word, level, "openjlpt"))
    return _write("ja", tiers)


_DOWNLOADERS = {"en": download_en, "zh": download_zh, "ja": download_ja}


def main() -> None:
    p = argparse.ArgumentParser(description="Download graded word lists.")
    p.add_argument("--language", default="all", choices=[*_DOWNLOADERS, "all"])
    args = p.parse_args()
    langs = list(_DOWNLOADERS) if args.language == "all" else [args.language]
    for lang in langs:
        counts = _DOWNLOADERS[lang]()
        print(f"{lang}: baseline={counts['baseline']} advanced={counts['advanced']}")


if __name__ == "__main__":
    main()
