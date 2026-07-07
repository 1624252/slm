# Vocabulary sources & licenses

Each language has `data/vocab/<lang>/baseline.csv` and `advanced.csv` with columns
`word,tier,source`:

- **baseline** — common words assumed already known (the learner's `K`).
- **advanced** — graded "to-learn" words (the target pool `T`).

## Shipped in this repo (committed)

| Files | `source` value | What it is | License |
| --- | --- | --- | --- |
| `en/*.csv`, `zh/*.csv`, `ja/*.csv` | `curated-sample` | Small, hand-authored lists for offline use and tests. Not exhaustive. | Original to this repo (CC0) |

These exist so the pipeline and tests run with no downloads. Replace/extend them with the
fuller lists below for real data generation.

## Generating fuller lists

**Frequency bands (any language, no scraping).** Uses [`wordfreq`](https://github.com/rspeer/wordfreq)
(code MIT; bundled data under permissive licenses):

```bash
python -m islm.vocab.build_lists --language <lang> --from-frequency --overwrite
```

Writes `tier=freq, source=wordfreq`. This is the language-agnostic default.

## Graded lists (CEFR / HSK / JLPT)

Sourced from the web by the language-scout subagents and fetched on demand by the downloader.
Only lists whose license permits redistribution are committed; anything restrictive is fetched
locally at build time instead. Verified sources and exact licenses are recorded here as they are
integrated:

| Language | Scheme | Source | License | Status |
| --- | --- | --- | --- | --- |
| English | CEFR | _pending scout verification_ | — | curated-sample in use |
| Chinese | HSK | _pending scout verification_ | — | curated-sample in use |
| Japanese | JLPT | _pending scout verification_ | — | curated-sample in use |
