# Vocabulary sources & licenses

Each language has baseline (known) and advanced (to-learn) tiers as CSVs with columns
`word,tier,source`:

- `data/vocab/<lang>/baseline.csv`, `advanced.csv` — small **curated samples** shipped in the
  repo so the pipeline and tests run offline. Each row carries its `source`: `curated-sample`
  rows are original content (CC0); other rows are individual words with their CEFR level taken
  from the graded lists below (`cefr-j-1.5`, `octanove-c1c2-1.0`) and are covered by those
  sources' licenses. English `advanced.csv` spans B1–C2 (skewed to harder C1/C2 targets).
- `data/vocab/<lang>/baseline.full.csv`, `advanced.full.csv` — **downloaded** graded lists
  (git-ignored; run `python -m islm.vocab.download`). Loaders prefer these when present.
- `data/vocab/en/exam.csv` — committed **GRE/SAT/ACT** exam words (tier = the exam), for
  genuinely hard advanced targets the CEFR lists miss. `download.py` merges these into
  `advanced.full.csv`, and a dozen exemplars also live in the committed `advanced.csv`.

## Downloaded graded lists (`islm.vocab.download`)

All URLs were fetched and verified. Baseline/advanced tier split shown per language.

| Language | Scheme | Dataset (repo) | License (exact) | Tiers |
| --- | --- | --- | --- | --- |
| English | CEFR (+ exam) | CEFR-J Vocabulary Profile 1.5 + Octanove C1/C2 (`openlanguageprofiles/olp-en-cefrj`) + committed `exam.csv` | CEFR-J: free research **and** commercial use **with citation** (© Tono Lab, TUFS); Octanove: **CC-BY-SA-4.0**; exam words are factual data (`exam-curated`) | baseline A1–A2, advanced B1–C2 **+ GRE/SAT/ACT** |
| Chinese | HSK 3.0 | `ivankra/hsk30` | **MIT** | baseline HSK1–3, advanced HSK4–9 |
| Japanese | JLPT | `evanclan/OpenJLPT` | **CC-BY-SA-4.0** (levels from Jonathan Waller/tanos.co.uk CC-BY; glosses from JMdict/EDICT) | baseline N5–N4, advanced N3–N1 |

## Frequency baselines (`islm.vocab.build_lists --from-frequency`, any language)

| Dataset | License |
| --- | --- |
| `hermitdave/FrequencyWords` (OpenSubtitles) — `en_50k`, `zh_cn_50k`, `ja` | code MIT / **content CC-BY-SA-4.0** |

## Attribution & redistribution

- **CC-BY-SA-4.0** (Octanove, JLPT levels, FrequencyWords content): attribute the source and keep
  the ShareAlike license on any redistributed derivative.
- **CEFR-J**: cite "CEFR-J Wordlist, Tono Lab, Tokyo University of Foreign Studies (TUFS)".
- **FrequencyWords / OpenSubtitles**: link to `http://www.opensubtitles.org/` in reports.
- **JLPT** is not an official word list; OpenJLPT is a community compilation.
- The downloaded `*.full.csv` files are **git-ignored** to avoid re-licensing third-party data in
  this repo; fetch them on demand. Committed in-repo: the CC0 curated samples plus `exam.csv`
  (individual words + their exam label — factual data, not a copyrightable compilation).

## Avoided (license issues, flagged by scouts)

`google-10000-english` (NOASSERTION/LDC-restricted), `gigacool/hanyu-shuiping-kaoshi` (data
header says CC-BY-NC), and repos with no LICENSE file — not redistributable, not used.
