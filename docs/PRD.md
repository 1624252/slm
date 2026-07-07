# PRD — i+1 Story SLM

A comprehensible-input language-learning story generator built as a fine-tuned small model.

| Field | Value |
| --- | --- |
| Owner | Jiaying Song |
| Status | Draft v1.0 — planning / pre-build |
| Last updated | 2026-07-07 |
| Source docs | `docs/spec.md` (build brief), `docs/brainlift.md` (research) |
| Timebox | ~1 week (see Milestones) |

---

## 1. Summary

We are building a **small, fine-tuned language model that writes short language-learning
stories from comprehensible input**. Given a learner's known-vocabulary set and a few target
words to teach, the model writes a story that stays inside the known vocabulary, adds **at most
one new word per sentence**, makes each new word's meaning **inferable from context**, **reuses
target words** for spaced repetition, and stays **compelling** (a small humor/surprise beat on
the target word) — all without announcing that it is a lesson.

The point is not raw capability. It is **reliability of a constrained behavior** in a tiny,
cheap model. A well-prompted base model does not hold all of these lexical, pedagogical, and
formatting constraints at once (see §4, §11). We buy that reliability with **data**: distill
examples from a frontier teacher, hard-filter them with deterministic validators, and fine-tune
a small open model with QLoRA. **The dataset is the deliverable; the model is the dataset made
runnable.**

---

## 2. Problems it solves

1. **Generic LLMs and human tutors cannot reliably hold the i+1 budget.** Producing text that
   is simultaneously in-vocabulary, ≤1-new-word-per-sentence, inferable, recurring, and
   engaging is a hard multi-constraint instruction-following problem. Even 70B models need
   validators plus rewriting to get there (SRS-Stories); prompting alone drifts (IFEval).
2. **Graded readers are static and finite.** A learner burns through them and then faces a cliff
   into native text. This generates **unlimited, personalized, level-appropriate** input.
3. **Flashcard SRS is decontextualized and dull.** Words learned as isolated pairs fade fast.
   This embeds spaced repetition **inside stories**, where context teaches meaning and
   recurrence makes it durable.
4. **"Level-appropriate" usually means boring.** Krashen's bar is *compelling* input — so
   interesting the learner forgets they are reading a second language. We treat engagement as a
   requirement, not a bonus, while guarding against off-target distraction.

---

## 3. Goals and non-goals

**Goals**
- Instill one narrow behavior (the Behavior Spec, §4) reliably in a ≤4B open model.
- Ship the full loop: generate → validate → judge → fine-tune → evaluate → demo.
- Produce a base-vs-tuned results table proving the fine-tune beats the prompted base on
  spec adherence and robustness.

**Non-goals**
- Beating frontier models on raw benchmarks (wrong measure; see `spec.md` §"why this exists").
- Broad chatbot tutoring, grammar-rule instruction, or two-way conversation (§9 out of scope).
- Pretraining from scratch. "Train" here means supervised fine-tuning (QLoRA).

---

## 4. Behavior Spec (the gate)

This is the falsifiable pass/fail rubric a stranger could apply to any output. Everything
downstream (data generation, evaluation) serves it.

> **For a given known-vocabulary list `K`, target-word set `T`, and theme, the model returns a
> short story in the target language such that: (1) every word is in `K ∪ T` (or a trivially
> inferable proper noun); (2) no sentence introduces more than one word from `T`; (3) each
> introduced target word's meaning is inferable from its sentence's context; (4) every target
> word in `T` recurs at least 3 times across the story; and (5) the story is coherent and
> compelling, with any humor/surprise attached to a target word rather than to an off-topic
> detail. It never states that it is teaching or labels the target words.**

**Litmus test (why fine-tune instead of prompt).** A good prompt on the base model does *not*
do this reliably: it leaks out-of-vocabulary words, crams multiple new words into a sentence,
repeats words in thin contexts, or wanders into seductive detail. Reliability under these
constraints is exactly what a dataset buys and a prompt cannot guarantee (`spec.md` §"the whole
game"; brainlift Insight 10).

---

## 5. Theoretical foundation

The two most important experts anchor the whole design; the rest calibrate the numbers.

- **Stephen Krashen — the core theory.** Acquisition happens when the learner understands input
  a little beyond their level (`i+1`) with help from context, and when the **affective filter**
  is low. Compelling input ("so interesting you forget it's in another language") supplies the
  reading *volume* acquisition needs, and **narrow reading** (a series, recurring characters and
  topics) lowers the comprehension burden while recycling words. → Drives: the ≤1-new-word rule,
  the "never announce the lesson" rule, the engagement requirement, and the serialized story
  world. *(Krashen 1982, 2004; Krashen, Lee & Lao 2018.)*
- **Lester Loschky — the guardrail.** In his Japanese-learning experiment, the group with the
  **best moment-to-moment comprehension did not show better vocabulary/structure acquisition**.
  Feeling understood ≠ having learned. → Drives the central evaluation stance: **measure felt
  fluency and actual learning separately**, and never treat "the learner enjoyed it" as proof.
  This is why hard validators, not vibes, gate the data and the eval. *(Loschky 1994.)*

Supporting evidence (full detail in `docs/brainlift.md`):

| Expert | Contribution | Design consequence |
| --- | --- | --- |
| Paul Nation | 98% coverage ≈ 1 unknown word / 50 running words | Makes ≤1-new-word-per-sentence compatible with whole-text coverage; sets the 95%/98% targets |
| Stuart Webb | Informative context → better *meaning* learning | The sentence around a new word must carry strong semantic clues (inferability) |
| Waring & Takaki | Words met < ~8 times are not retained at 3 months | Recurrence floor: ≥3 in a story, ~8 across a series |
| R. Reed Hunt | Isolation (von Restorff) effect | One new word among known words already "pops" — keep the surroundings plain |
| Schmidt | Humour effect is near zero-sum | Reserve the funny beat for the target-word sentence only |
| Harp & Mayer | Seductive details reduce recall/transfer | Off-target engagement is a **defect**, not color |
| SRS-Stories (2025) | Direct precedent: prompting + validator-guided rewriting cut OOV ~6.7% → ~0.6%; constrained decoding was worst | Our pipeline shape (generate → validate → rewrite), and the open gap we fill: distill into a *small* model |
| IFEval (2023) | LLMs struggle with verifiable multi-constraint instructions | Justifies fine-tuning + rule-based checks over clever prompting |

**Known tension (documented, not hidden):** Krashen's "rough tuning beats fine tuning" argues
*against* engineering exactly one new word per sentence — he would flood the learner with
abundant input and let `i+1` emerge. Our bet is that a small model can *control* `i+1` without
overshooting while still producing enough volume and interest. We revisit this in §16 (risks).

---

## 6. Target user and persona

Start niche, then widen. The shipped languages are **English, Chinese (Mandarin), and
Japanese**; the pipeline is **language-agnostic** (any other language falls back to frequency
bands + a generic tokenizer).

**Primary persona — "Maya," the intermediate self-studier.**
- Adult, self-directed learner of **English, Chinese, or Japanese** as a second language.
- Roughly **intermediate** (CEFR A2→B1 / HSK 3–4 / JLPT N4→N3): can read and form simple
  sentences, so there is a vocabulary base to build on (per `notes.txt`).
- Wants to grow vocabulary by **reading things she enjoys**, not grinding flashcards.
- Frustrated that "graded" content is either too hard (native text) or too dull.
- Reads on a phone/laptop; values short sessions and a sense of momentum.

**Secondary persona — the builder/researcher (you).** Needs the data, validators, and eval
harness to be reproducible and publishable (HF Hub).

**Domain boundary:** English, Chinese, and Japanese, each with a baseline (known) tier and a
graded advanced (to-learn) tier, in a serialized story world (narrow reading). Languages beyond
these three are supported by the architecture but not curated (§9).

---

## 7. User stories

**Focused on (MVP):**
- As a learner, I want a story I can *almost fully* understand, so that I stay in flow and keep
  reading. *(compelling input; 98% coverage)*
- As a learner, I want new words introduced gently and guessable from context, so that I never
  feel like I'm failing. *(≤1 new word/sentence; inferability; affective filter)*
- As a learner, I want the new words to come back within and across stories, so that they
  actually stick. *(recurrence; spaced repetition)*
- As a learner, I want the story to be fun or surprising, so that it's memorable — but not so
  wild I lose the thread. *(humor on-target; no seductive detail)*
- As a learner, I want to give my level or known-word list and the words I want to learn, so the
  story is tuned to me. *(controllable i+1)*

**Not focused on (deferred):**
- As a learner, I want to chat back and forth with a tutor (interactive negotiation of meaning).
- As a learner, I want audio / pronunciation / images.
- As a learner, I want the system to auto-learn my vocabulary from my past writing.
- As a learner, I want explicit grammar lessons and rule drills.
- As a learner, I want many languages on day one.

---

## 8. MVP definition

**In scope**
- A fine-tuned ≤4B model that, from `(language, K, T, theme)`, emits one spec-compliant story in
  the target language (English, Chinese, or Japanese).
- Deterministic **validators** (coverage/OOV, ≤1-new-word, recurrence, inferability proxy).
- A **data-generation pipeline** (teacher distillation + validator-guided rewrite + judge gate).
- A published **dataset** (JSONL) and the **eval harness** with a base-vs-tuned results table.
- A minimal **inference demo** (CLI or Gradio) that takes a level + target words and returns a
  validated story.

**Out of scope (MVP)**
- Persistent learner accounts, long-horizon SRS scheduling across sessions, and vocabulary
  inference from usage (schema is defined in §13 but not the product surface).
- Languages beyond the shipped English/Chinese/Japanese (the design is language-agnostic, so
  adding one is a data task, not a redesign); audio/vision; and conversational tutoring.
- A polished web app; the demo is intentionally thin.

---

## 9. Functional requirements

1. **Input.** Accept a learner level (maps to a `K` list) or an explicit `K`; a target set `T`
   (1–5 words); an optional theme/series id.
2. **Generation.** Produce a story of ~8–15 sentences using only `K ∪ T` (+ inferable proper
   nouns).
3. **Constraint enforcement at inference.** Run validators on the output; if it fails, run a
   bounded rewrite/regenerate loop (≤5 passes, per SRS-Stories) before returning.
4. **Recurrence.** Ensure each `T` word appears ≥3 times; expose per-word counts.
5. **Transparency for eval (not for the learner).** Emit metadata (OOV rate, new-words/sentence,
   recurrence, sentence count) alongside the story for logging and scoring.
6. **Determinism knob.** Support temperature 0 for reproducible eval runs.

---

## 10. System architecture

```
level / K / T / theme
        │
        ▼
[1] Scenario sampler ──► [2] Teacher generation (frontier LLM)
                                   │
                                   ▼
                         [3] Deterministic validators
                         (tokenize→lemmatize→coverage,
                          ≤1-new-word, recurrence)
                                   │  fail
                                   ▼
                         [4] Validator-guided rewrite (≤5 passes)
                                   │  pass
                                   ▼
                         [5] LLM-as-judge quality gate
                         (inferability, engagement, coherence,
                          seductive-detail control)
                                   │  keep
                                   ▼
                         [6] Dataset (JSONL, train/val/test)
                                   │
                                   ▼
                         [7] QLoRA SFT on small open model
                                   │
                                   ▼
                         [8] Eval harness (base vs tuned)
                                   │
                                   ▼
                         [9] Inference demo (+ inference-time validators)
```

Steps [3]–[5] are reused at both **data-generation time** (to build the training set) and
**inference time** (to guard the shipped model). The validators are the backbone: the same code
that filters data also grades outputs and guards inference.

---

## 11. Base model and tech stack

**Base model.**
- **Primary: `Qwen3-4B-Instruct`.** Chosen for strong **multilingual** coverage (notably
  English, Chinese, and Japanese), an Instruct variant for fast SFT, and a size that fits a
  single 24 GB GPU under QLoRA — matching `spec.md`'s stack guidance (Qwen3 0.6B/1.7B/4B).
- **Smaller fallback / edge target: `Qwen3-1.7B-Instruct`** (and `0.6B` as an on-device stretch)
  to test how small the behavior can go — the interesting research question (cf. SRS-Stories used
  a 70B; we ask whether it distills to ≤4B).
- **Alternates if needed:** Llama 3.2 1B/3B, Gemma 3 small, SmolLM3.

**Teacher & judge (distillation).**
- **Teacher:** a frontier model for generation (e.g., a GPT-5-class / Claude / Gemini 3 /
  Qwen3-235B model — AI costs are covered per `spec.md`).
- **Judge:** a *different* frontier model than the teacher, to reduce self-preference bias
  (SRS-Stories used Llama-3.1-70B generator + Qwen2.5-72B judge).

**Tooling.**

| Concern | Choice |
| --- | --- |
| Fine-tuning | Unsloth + QLoRA (≈2× faster, ≈70% less VRAM); TRL/PEFT underneath |
| Compute | Single A100/H100 via Colab / Modal / RunPod (`notes.txt`: Colab) |
| Tokenize/lemmatize | spaCy (English), jieba (Chinese), fugashi+UniDic (Japanese); generic Unicode fallback for other languages |
| Word lists | CEFR (English), HSK (Chinese), JLPT (Japanese); `wordfreq` frequency bands for any language |
| Data format | JSONL (chat-format SFT); preference pairs for optional DPO |
| Eval | LLM-as-judge + deterministic validators; pandas for the results table |
| Tracking | Weights & Biases (optional) or CSV |
| Lint / format | `ruff` (lint + format) |
| Tests | `pytest` for validators (a validator bug corrupts both data and eval) |
| Publish | Dataset + model to Hugging Face Hub; demo via Gradio/Spaces |
| Env | Python 3.11, `requirements.txt` |

---

## 12. Dataset — the deliverable

**Task formalization (from SRS-Stories).** Given a known set `K` and to-learn set `T`, generate
a coherent story using (almost) only `K ∪ T`, with each `T` word appearing ≥3 times. This is the
unit the dataset teaches.

**Sourcing (`notes.txt`).**
1. **Word lists & levels:** graded lists per language — CEFR (English), HSK (Chinese), JLPT
   (Japanese) → baseline `K` banks and advanced `T` pools; `wordfreq` frequency bands as a
   universal fallback for any language.
2. **Hugging Face:** existing graded-reader / story / CEFR datasets and any released SRS-Stories
   assets, used as seeds and style references.
3. **Generation (primary):** distill from the frontier teacher — this is where most examples
   come from. The craft is in the generation prompt + quality gate, not raw volume.
4. **Small human-labeled gold set:** a few dozen hand-checked stories reserved for eval and
   judge-calibration.

**Generation pipeline (per example).**
1. **Sample a scenario:** level band → `K` subset; pick 1–5 `T` words (mixing frequency bands);
   pick a theme/series slot; specify a target-carrying humor/surprise beat.
2. **Generate** with the teacher using a detailed prompt that encodes the full Behavior Spec.
3. **Validate** deterministically: normalize → tokenize → lemmatize → check every token against
   `K ∪ T`; compute OOV rate, max new-words-per-sentence, per-`T` recurrence, sentence lengths.
4. **Rewrite loop** (≤5 passes) to drive OOV → 0 and fix any ≥2-new-word sentences. **No hard
   constrained decoding** — SRS-Stories found it the *worst* method (ungrammatical, off-context).
5. **Judge gate:** score inferability, compellingness, coherence, and seductive-detail control
   (humor must carry a target word); reject "announced the lesson" outputs.
6. **Keep** only examples passing **both** the hard validators and the judge threshold. Target a
   few hundred to a few thousand high-quality examples.

**Splits.** Split at the **scenario level** (level × theme × target-word combo) so no scenario
leaks between train/val/test.

**Preference data (optional, for DPO stretch).** For a kept story (chosen), synthesize an
off-spec variant (rejected) — e.g., inject one OOV word, or a second new word in a sentence, or
swap the on-target joke for a seductive detail — to sharpen adherence beyond SFT.

---

## 13. Data schema (what we store, where)

MVP storage is **flat files** (JSONL datasets on disk, published to the HF Hub; CSV eval
results). The learner-profile / SRS store is **specified now** but only becomes a live database
in a later version (candidate: SQLite).

**13.1 Training example (JSONL, one record per line):**

```json
{
  "id": "en-A2-0001",
  "language": "en",
  "level": "A2",
  "theme": "detective-cat-series/ep03",
  "known_vocab_ref": "cefrj:A2",
  "target_words": ["whisker", "clue"],
  "messages": [
    {"role": "system", "content": "<generation spec / rules>"},
    {"role": "user", "content": "Known level: A2. Teach: whisker, clue. Theme: detective cat, ep 3."},
    {"role": "assistant", "content": "<the story>"}
  ],
  "metadata": {
    "sentences": 12,
    "oov_rate": 0.0,
    "max_new_words_per_sentence": 1,
    "target_recurrence": {"whisker": 3, "clue": 4},
    "judge_scores": {"spec_adherence": 2, "inferability": 2, "engagement": 2, "coherence": 2},
    "teacher_model": "<teacher-id>",
    "rewrite_passes": 2,
    "split": "train"
  }
}
```

**13.2 Vocabulary store (per language):** `word → { lemma, cefr_level, freq_rank, pos }`.
Stored per language at `data/vocab/<lang>/{baseline,advanced}.csv` (`word,tier,source`), derived
from CEFR/HSK/JLPT + `wordfreq`; used by the validators and the scenario sampler.

**13.3 Learner profile + SRS store (future SQLite schema):**

| Table | Key columns | Purpose |
| --- | --- | --- |
| `learner` | `id`, `language`, `level` | Who the stories are for |
| `known_word` | `learner_id`, `word`, `source` | The evolving `K` set |
| `target_word` | `learner_id`, `word`, `status`, `encounters`, `last_seen`, `next_due` | SRS state; enforces the ~8-encounter floor across sessions |
| `story` | `id`, `learner_id`, `theme`, `created_at`, `metadata_json` | Generated stories + their validator metadata |

**13.4 Eval result (CSV / JSONL):** one row per (scenario × model) with the hard-check
booleans/rates and judge scores (see §14), for the base-vs-tuned table.

---

## 14. Evaluation (built before training)

Per `spec.md`, **no training before the eval exists.** Evaluation is the make-or-break piece.

**14.1 Held-out scenario set.** 50–100 scenarios `(level, K, T, theme)`, scenario-split from
training data, plus the human-labeled gold subset.

**14.2 Deterministic behavioral checks (the failures the spec forbids).**

*OOV = out-of-vocabulary (a word not in `K ∪ T`); OOV rate = OOV ÷ total words; coverage = 1 − OOV rate.*

| Check | Definition | Target (tuned) |
| --- | --- | --- |
| Coverage / OOV | % running words outside `K ∪ T` (out-of-vocabulary) | ideal 100% coverage; gate OOV ≤ 2% (coverage ≥ 98%) |
| ≤1-new-word rule | max new `T` words in any sentence | ≤ 1 in 100% of sentences |
| Recurrence | each `T` word appears ≥ 3× | satisfied for ≥ 90% of target words |
| Inferability proxy | cloze: mask the target word, ask a judge/model to recover it from context | ≥ 60% recovered |

**14.3 LLM-as-judge rubric** (extends `spec.md` Appendix A; each 0/1/2):
Spec adherence, Robustness, Task quality (coherence + compellingness), Consistency — plus
project-specific **Inferability** and **Seductive-detail control** (humor on-target, lesson not
announced). Report mean per dimension.

**14.4 Base-vs-tuned comparison.** Run identical scenarios through the **prompted base** and the
**tuned** model at temperature 0; report hard-check pass rates and judge means side by side with
the **delta**. **Win condition (from `spec.md`): the tuned model beats the base on Spec
adherence and Robustness.**

**14.5 Robustness / adversarial (stretch).** Scenarios designed to break it — tiny `K`, rare
`T`, jargon-tempting themes — to measure OOV blow-up under pressure (Behavior Spec robustness).

**14.6 Human-in-the-loop.** SRS-Stories reports only **moderate** human–judge correlation
(English r ≈ 0.46–0.56, lower for other languages). Therefore: **hard validators are the primary
gate**, the LLM judge is secondary, and humans spot-check the final set (echoing Loschky:
comprehension ≠ acquisition, so we never trust felt quality alone).

---

## 15. Success criteria

- Tuned model **beats prompted base on Spec adherence and Robustness** (primary, per spec).
- OOV ≤ 2% (coverage → 100%), ≤1-new-word in 100% of sentences, recurrence ≥ 90% — all **higher than base**.
- A reproducible results table and error-analysis paragraph naming where the tuned model still
  fails and whether it's a data problem.

---

## 16. Milestones (maps to the one-week arc in `spec.md`)

| Day | Milestone | Checkpoint |
| --- | --- | --- |
| 1 | Env runs base inference; brainlift + spiky POVs done | Base model responds; behavior known |
| 2 | Behavior Spec finalized; validators + eval harness + data-gen pipeline; 50 junk examples | Full loop runs end to end |
| 3 | v1 dataset generated + filtered; first QLoRA run; first base-vs-tuned eval | Midweek gate: numbers on the board |
| 4 | Diagnose failure modes; **fix in data, not hyperparameters**; retrain | One failure mode resolved via data |
| 5 | Final eval + error analysis; ship demo; record 3–5 min video | Submission package ready |

**Stretch ladder** (from `spec.md`, in order): (1) DPO on on-spec vs off-spec pairs; (2)
adversarial/robustness eval; (3) composed behavior (e.g., hold i+1 *and* a fixed reading level
*and* stay encouraging).

---

## 17. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Tokenizer/segmenter errors inflate/deflate OOV (esp. Chinese/Japanese) | Use jieba (zh) + fugashi/UniDic (ja); match on lemma OR surface; unit-test per language; manual audit a sample |
| LLM-judge unreliability (moderate correlation) | Hard validators primary; humans for final; different judge vs teacher |
| Small model can't hold all constraints | Scale 1.7B→4B; inference-time validate+rewrite loop; DPO stretch |
| Scope creep to many languages | Curate EN/ZH/JA; any other language degrades gracefully to frequency bands + a generic tokenizer |
| Data leakage between splits | Scenario-level splitting (level × theme × target combo) |
| Krashen's "rough tuning" objection to engineered i+1 | Documented bet: controllable i+1 + volume; eval measures whether it holds |
| "Learn vocab from usage" is hard | Deferred; schema defined but not built for MVP |
| Over-funny stories become seductive details | Judge dimension for on-target humor; keep surroundings plain (Hunt/Schmidt) |

---

## 18. Repository structure

```
slm/
├── README.md
├── .gitignore
├── .env.example               # LLM keys template (.env is git-ignored)
├── requirements.txt
├── pyproject.toml             # package metadata + ruff + pytest config
├── docs/
│   ├── PRD.md                 # this document
│   ├── dataset-and-eval.md    # how the dataset + eval work and how to run them
│   ├── spec.md                # build brief
│   └── brainlift.md           # research base
├── src/islm/
│   ├── config.py              # thresholds + LLM settings
│   ├── vocab/                 # languages, tokenize, analyzers (en/zh/ja/generic), word lists, build_lists
│   ├── validators/            # coverage, ≤1-new-word, recurrence -> ValidationReport
│   ├── llm/                   # OpenAI-compatible client, offline MockLLM, prompts
│   ├── datagen/               # scenario sampler, generate + rewrite loop, pipeline CLI
│   ├── eval/                  # LLM judge, cloze, base-vs-tuned harness + report
│   ├── train/                 # (planned) Unsloth/QLoRA scripts + configs
│   └── infer/                 # (planned) inference + inference-time validators + demo
├── tests/                     # validator unit tests + offline end-to-end smoke test
├── data/
│   ├── vocab/                 # bundled sample word list (tracked)
│   └── generated/             # datasets (git-ignored; published to HF Hub)
└── evals/
    ├── scenarios/             # held-out scenarios (tracked, reproducible)
    └── results/               # base-vs-tuned outputs (git-ignored)
```

---

## 19. Deliverables (final submission package, per `spec.md`)

1. The **dataset**, published (the real artifact).
2. The **model on the Hugging Face Hub** + a running inference demo.
3. **Eval harness + results table** — base vs tuned, with the behavior metrics above.
4. **Brainlift** — the behavior thesis and whether data→behavior held, with evidence
   (`docs/brainlift.md`).
5. **3–5 min demo video** showing the model doing what the base model fails to do reliably.

---

## 20. Open questions

- Which exact CEFR-J band and how large a `K` per level give the best coverage/interest balance?
- How small can we go (4B → 1.7B → 0.6B) before spec adherence collapses?
- Series design: how many recurring themes are needed to hit the ~8-encounter series floor
  without feeling repetitive?
- Teacher/judge model choice under the covered-cost constraint, and judge-calibration against
  the human gold set.
