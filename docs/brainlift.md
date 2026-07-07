# BrainLift: Small Learning Model for i+1 Language Stories

## Owners

- Jiaying Song

## Purpose

### Purpose

The purpose of this BrainLift is to collect research for a small learning model that writes language-learning stories using comprehensible input. The stories should stay inside a learner's known vocabulary, add at most one new word per sentence, make the new word understandable from context, and use humor or surprise without letting the joke take over the lesson.

### In Scope

- Krashen's Input Hypothesis, `i+1`, comprehensible input, and the affective filter.
- Lexical coverage, unknown-word density, word-frequency lists, CEFR vocabulary lists, and contextual inference.
- Repetition, retrieval, spacing, and narrow reading for vocabulary retention.
- Compelling input, humor, surprise, distinctiveness, attention, and seductive-detail risks.
- Dataset generation, constrained SFT, QLoRA, automatic lexical validators, LLM-as-judge scoring, and base-vs-tuned evaluation.
- Instruction-following limits in LLMs, especially strict lexical, formatting, and multi-constraint adherence.

### Out of Scope

- Broad chatbot tutoring not tied to controlled story generation.
- Raw benchmark capability as the measure of project success.
- Interactive, two-way negotiation of meaning, unless the tool later becomes conversational.
- Raw grammar-syllabus or explicit-rule instruction as the primary teaching mechanism.

---

## DOK 4: Spiky Points of View (SPOVs)

- **Spiky POV 1:** Language learning works best when the learner does not realize they are being taught. In other words, trick them into it.
  - **Elaboration:** Vocabulary acquisition should feel like reading for fun, not doing a lesson. If the story announces the target word, the unknown word starts to feel like failure (Insight 1). If the word sits inside 98% familiar input, the learner can feel fluent while still meeting `i+1` material (Insight 3). One exposure will not do much, so the learner needs a reason to keep reading until the word comes back again and again (Insight 4). The trick belongs in the experience, not the measurement. Evaluation still has to check coverage, recurrence, inferability, and learning gains (Insight 9).
- **Spiky POV 2:** A constrained language model can be a more efficient language partner than a human.
  - **Elaboration:** A human conversation partner is useful, but not great at controlled `i+1` input. Most people cannot track a learner's known vocabulary, hold the unknown-word budget, recycle target words, and make every new meaning inferable in real time. A language model can, if it is trained and checked against those rules (Insights 3, 5, and 10). It also lowers the social cost of practice. The learner can ask for clarification, repetition, or rephrasing without worrying about sounding foolish, which matters because anxiety and low self-esteem raise the affective filter (Insight 1). Loschky's negotiated-interaction result still warns that comprehension alone is not enough, but an AI partner can make negotiation cheaper, more frequent, and easier to measure (Insights 2 and 9).

---

## Experts

- **Stephen D. Krashen**
  - **Who:** Emeritus Professor of Education, University of Southern California.
  - **Focus:** Input Hypothesis, Comprehension Hypothesis, affective filter, compelling input, narrow reading, and free voluntary reading.
  - **Why Follow:** Primary source for the project's core theory: acquisition through comprehensible input slightly above the learner's current level. His work on compelling input and narrow reading shapes the story design.
  - **Where:** [sdkrashen.com](http://sdkrashen.com/)
- **Lester C. Loschky**
  - **Who:** Professor of psychological sciences at Kansas State University; researcher on second-language acquisition, attention, and visual cognition.
  - **Focus:** Comprehensible input, negotiated interaction, and the relationship between moment-to-moment comprehension and acquisition.
  - **Why Follow:** His Japanese-learning experiment warns that better immediate comprehension does not automatically mean better vocabulary or structure acquisition.
  - **Where:** [Kansas State SLA publications](https://www.k-state.edu/psych/vcl/publications/second-language-acquisition-publications.html)
- **Paul Nation**
  - **Who:** Emeritus Professor of Applied Linguistics, Victoria University of Wellington.
  - **Focus:** Vocabulary acquisition, lexical coverage, graded readers, vocabulary size, and frequency-based word lists.
  - **Why Follow:** Gives the coverage thresholds (95% and 98%) and word-family lists that turn "known vocabulary" into something measurable.
  - **Where:** [VUW profile](https://people.wgtn.ac.nz/paul.nation)
- **Stuart Webb**
  - **Who:** Professor of Applied Linguistics, Western University.
  - **Focus:** Incidental vocabulary learning and context quality.
  - **Why Follow:** His experiments show why the sentence around a new word must make the meaning inferable, not merely include the word.
  - **Where:** [Western University profile](https://www.edu.uwo.ca/about/faculty-profiles/stuart-webb/index.html)
- **Rob Waring**
  - **Who:** Extensive-reading and graded-reader researcher, Notre Dame Seishin University.
  - **Focus:** Extensive reading, graded readers, and vocabulary retention over time.
  - **Why Follow:** His graded-reader studies show how quickly gains from one story fade, which is why target words need to return across a series.
  - **Where:** [Academia.edu](https://ndsu-jp.academia.edu/RobWaring)
- **Richard E. Mayer**
  - **Who:** Distinguished Professor of Psychology, UC Santa Barbara.
  - **Focus:** Multimedia learning, the coherence principle, and cognitive load.
  - **Why Follow:** His seductive-details and coherence work helps set the boundary between an engaging story and a distracting one.
  - **Where:** [UCSB profile](https://psych.ucsb.edu/people/faculty/richard-mayer)
- **R. Reed Hunt**
  - **Who:** Professor of Psychology, University of Mississippi; leading researcher on distinctiveness and the isolation effect.
  - **Focus:** Distinctiveness in memory, the von Restorff (isolation) effect, and why distinctive items are better recalled.
  - **Why Follow:** Explains why a single new word set against familiar vocabulary "pops" and is more likely to be remembered.
  - **Where:** [University of Mississippi profile](https://olemiss.edu/directory/reed-hunt/)

---

## DOK 3: Insights

### Why hiding the lesson works better than announcing it

- **Insight 1 (the announcement is the obstacle):** The affective filter is about more than test anxiety. Labeling something as a vocabulary lesson can raise the filter by making the unknown word feel like a small failure. Krashen argues that high filters block acquisition even when the message is understood, while compelling input lowers anxiety by moving attention back to meaning. The normal "here is a word, learn it" framing works against the goal. It turns easy reading into study.
- **Insight 2 (feeling understood != having learned):** Loschky found that negotiated interaction produced the best moment-to-moment comprehension, but that comprehension did not correlate with vocabulary or structure gains. That makes the learner's *sense* of understanding a weak measure of learning. A learner can feel like they "get it" while acquisition is happening slowly underneath, so the product needs to track felt fluency and measured learning separately.

### How comprehensible input becomes acquired vocabulary

- **Insight 3 (the 98% comfort zone is what hides the new word):** Nation's coverage math shows that 98% coverage is roughly one unknown word in 50 running words. The one-new-word-per-sentence rule fits inside that budget. The learner can meet the new word without hitting an "I don't understand" moment because the rest of the sentence is familiar.
- **Insight 4 (one clever sentence teaches almost nothing; volume does):** Nagy, Herman, and Anderson show that contextual learning is small per encounter, and Waring & Takaki show that gains from one graded story fade unless words come back many times. The product's unit of value is not one brilliant sentence. It is repeated exposure over time, which only happens if the learner wants to keep reading.
- **Insight 5 (context teaches meaning, repetition makes it durable):** Webb shows that richer context improves *meaning* knowledge, while Waring & Takaki show that unrepeated story encounters fade quickly. A story generator that writes one vivid sentence leaves the word fragile. A generator that repeats the word in thin contexts leaves meaning under-learned. The design needs both a strong clue in the sentence and recurrence across the series.

### Making the target memorable without breaking the spell

- **Insight 6 (make the word distinctive, but only the word):** Hunt's isolation effect and Schmidt's humour effect point to the same useful limit. A single new word surrounded by familiar vocabulary is already an isolation setup, so it is more likely to be remembered. The problem is that humor and surprise lose their force when every sentence tries to be funny or strange. The target word should get the vivid moment. The surrounding language should stay plain.
- **Insight 7 (surprise should sharpen learning, not hijack it):** The story has to be interesting enough that the learner wants to keep reading. Krashen, Lee & Lao call this "compelling input." A funny or surprising beat can help the target word stand out (Hunt; Schmidt), but only if the beat clarifies the word's meaning. If the learner remembers the twist instead of the word, the twist has become a seductive detail. Harp & Mayer show that those details reduce recall and transfer. Aim for a small, target-carrying surprise, not a wild plot detour.
- **Insight 8 (off-target engagement is a defect):** Harp & Mayer show that interesting but irrelevant "seductive details" reduce recall of main ideas and transfer, and that obvious fixes do not solve the problem. Humor, surprise, and emotion are useful only when they carry the target word's meaning. A funny flourish that does not help the word is a bug, not a bonus.

### Building and trusting the system

- **Insight 9 (trick awareness, never trick the evaluation):** Because comprehension does not equal acquisition (Insight 2), the system cannot treat "the learner enjoyed it and felt fluent" as proof of learning. The experience can feel effortless and lesson-free. The measurement cannot. It needs lexical validators for coverage and the <=1-new-word rule, recurrence tracking for the ~8+ encounter floor, and separate scoring for inferability and engagement.
- **Insight 10 (prompting will not reliably hold all the constraints):** The behavior spec asks for known vocabulary, at most one new word per sentence, inferable meaning, recurrence, and engaging story quality at the same time. IFEval and FollowBench show that LLMs still struggle with precise, verifiable, multi-constraint instructions. SRS-Stories shows that this exact story task needs validation and rewriting to push out-of-vocabulary words down. A clever prompt is not enough. The safer path is a small model trained on filtered examples, backed by deterministic validators and rewrite/regenerate loops.

---

## DOK 2: Knowledge Tree

The Knowledge Tree below contains DOK 1 facts and DOK 2 summaries only. Facts come from primary sources. When a widely cited number belongs to a companion paper, that is noted rather than folded into the wrong citation.

- **Comprehensible Input And Second Language Acquisition**
  - **The Input Hypothesis And i+1**
    - **Source: Krashen (1982), *Principles and Practice in Second Language Acquisition*.**
      - **DOK 1 - Facts:**
        - The Input Hypothesis claims a necessary (not sufficient) condition to move from stage `i` to stage `i+1` is that the acquirer understand input containing `i+1`, where "understand" means focused on meaning, not form.
        - We acquire only when we understand language "a little beyond" our current level, with the help of context and extra-linguistic information (knowledge of the world).
        - Input need not contain *only* `i+1`; if the learner understands and there is enough input, "`i+1` will be provided automatically." Krashen argues that rough tuning through abundant comprehensible input beats deliberate fine tuning to one target.
        - Acquisition (subconscious, initiates fluent utterances) and learning (conscious, acts only as a Monitor/editor) are distinct, independent systems.
        - A high affective filter (anxiety, low motivation, low self-esteem) blocks input from reaching the language acquisition device even when the message is understood.
      - **DOK 2 - Summary & Analysis:**
        - Krashen gives the project its main theory, but his work also complicates the product's central rule. His "rough tuning beats fine tuning" position argues *against* engineering exactly one new word per sentence. He would supply abundant comprehensible input and let `i+1` emerge. The design bet here is different: a small model can control `i+1` without overshooting the learner, while still producing enough volume and interest for acquisition to happen.
      - **Link to source:** [Krashen 1982](http://sdkrashen.com/content/books/principles_and_practice.pdf)
    - **Source: Krashen (2004), "Applying the Comprehension Hypothesis: Some Suggestions," *IJFLT*.**
      - **DOK 1 - Facts:**
        - Acquisition proceeds faster when input is "narrow": a lot of input in a narrow range of topics or by a single author. The learner can reuse background knowledge, while broad "survey" exposure often stays incomprehensible.
        - Narrow reading is supported by case studies (e.g., adult ESL readers progressing through a single book series).
        - Narrow listening (Dupuy 1999): intermediate learners went from about half to nearly full comprehension after three to four listenings of same-topic recordings.
        - The goal is "autonomous acquirers" who understand that progress comes from comprehensible input, not from grammar study and vocabulary lists.
      - **DOK 2 - Summary & Analysis:**
        - Narrow reading points toward a *series*, not one-off stories. Recurring characters, settings, and topics lower the comprehension burden and make target words recur naturally. A serialized story world can deliver repeated encounters without feeling like drill.
      - **Link to source:** [Krashen 2004](https://www.sdkrashen.com/content/articles/2004_applying_the_comprehension_hypothesis_krashen.pdf)
  - **Comprehension, Negotiated Interaction, And Their Limits**
    - **Source: Loschky (1994), "Comprehensible Input and Second Language Acquisition: What Is the Relationship?" *Studies in Second Language Acquisition*.**
      - **DOK 1 - Facts:**
        - Experiment with learners of Japanese, comparing unmodified input, premodified input, and unmodified input with the chance for negotiated interaction.
        - Moment-to-moment comprehension was highest for the negotiated-interaction group.
        - There was no correlation between moment-to-moment comprehension and gains in vocabulary recognition or acquisition of target structures, though all groups showed significant gains.
      - **DOK 2 - Summary & Analysis:**
        - Negotiated interaction means speakers repair comprehension problems through clarification, confirmation, repetition, or rephrasing. Loschky's result separates "I understood it right now" from "I acquired it." For this product, a story that feels comprehensible still needs independent checks for recurrence, inferability, and learning. Fluent reading is not proof of acquisition.
      - **Link to source:** [Loschky 1994 (DOI)](https://doi.org/10.1017/S0272263100013103)
- **Vocabulary Control, Context, And Retention**
  - **Lexical Coverage (How Much Must Be Known)**
    - **Source: Nation (2006), "How Large a Vocabulary Is Needed for Reading and Listening?" *Canadian Modern Language Review*.**
      - **DOK 1 - Facts:**
        - 98% coverage of a written text requires about 8,000 to 9,000 word families; spoken text requires about 6,000 to 7,000.
        - A worked example (9,000 families plus proper nouns) reaches 98.24% coverage, or "one unknown word in about every 50 running words."
        - Coverage is heavily front-loaded: the first 1,000 families supply roughly 78 to 81% of written text, the second 1,000 another 8 to 9%.
        - Figures derive from fourteen 1,000-family lists built on the 100-million-token British National Corpus.
      - **DOK 2 - Summary & Analysis:**
        - This supports the "one new word per sentence" rule. At 98% coverage there is roughly one unknown word per 50 running words, so a 10 to 15 word sentence with one new word can still fit the coverage budget. The sentence-level rule and whole-text coverage are compatible. The front-loaded nature of coverage also tells the model which known-word bands to draw from.
      - **Link to source:** [Nation 2006 (open PDF)](https://www.lextutor.ca/cover/papers/nation_2006.pdf)
  - **Learning Words From Context**
    - **Source: Nagy, Herman & Anderson (1985), "Learning Words from Context," *Reading Research Quarterly*.**
      - **DOK 1 - Facts:**
        - 57 eighth-graders read ~1,000-word natural texts with target words and were tested with measures tapping partial word knowledge.
        - A single natural-context encounter produced small but statistically reliable gains in word knowledge.
        - The often-quoted ~0.05 probability of learning a word per encounter, about one in twenty, comes from the companion paper (Nagy, Anderson & Herman, 1987), and should be cited there.
      - **DOK 2 - Summary & Analysis:**
        - One clever sentence is not enough to teach a word. Contextual learning is real, but weak per encounter, and it accumulates through lots of reading. The model cannot rely on a single striking sentence. It needs many meaningful encounters across the story and the series.
      - **Link to source:** [Nagy, Herman & Anderson 1985 (DOI)](https://doi.org/10.2307/747758)
    - **Source: Webb (2008), "The Effects of Context on Incidental Vocabulary Learning," *Reading in a Foreign Language*.**
      - **DOK 1 - Facts:**
        - 50 Japanese EFL learners met disguised nonwords in either more-informative or less-informative single-sentence contexts.
        - More-informative contexts produced far better meaning knowledge (recall 1.31 vs 0.13; recognition 6.77 vs 4.38).
        - Context quality had no significant effect on knowledge of word *form*.
      - **DOK 2 - Summary & Analysis:**
        - Webb's experiment is the closest match to the "meaning inferable from context" rule. Context quality drives meaning learning, so the sentence around a new word needs strong semantic clues. The model cannot count on the learner guessing from weak context. Because form is learned through repeated exposure, the generator needs both rich single-sentence clues and recurrence.
      - **Link to source:** [Webb 2008 (archived)](https://web.archive.org/web/20240908223254/http://www2.hawaii.edu/~readfl/rfl/October2008/webb/webb.html)
  - **Repetition, Spacing, And Retention**
    - **Source: Waring & Takaki (2003), "At What Rate Do Learners Learn and Retain New Vocabulary From Reading a Graded Reader?" *Reading in a Foreign Language*.**
      - **DOK 1 - Facts:**
        - 15 learners read a 400-headword graded reader (~5,900 running words, 96.2% coverage) containing 25 target words at varying occurrence frequencies; tested immediately, after one week, and after three months.
        - Meaning knowledge fell from 4.6/25 (17.6%) immediately to 0.9/25 (3.6%) at three months, roughly one word's meaning retained.
        - Words met fewer than about eight times were essentially not retained three months later.
      - **DOK 2 - Summary & Analysis:**
        - One graded story is not enough for durable learning. Retention collapses without re-exposure. Target words need to reappear across multiple stories and sessions, and the "~8 encounters" floor is the minimum the series scheduler should guarantee for any word it intends to teach.
      - **Link to source:** [Waring & Takaki 2003](https://www2.hawaii.edu/~readfl/rfl/October2003/waring/waring.html)
- **Story Engagement, Memory, And Attention**
  - **Compelling And Narrative Input**
    - **Source: Krashen, Lee & Lao (2018), *Comprehensible and Compelling* / "The Compelling (not just interesting) Input Hypothesis."**
      - **DOK 1 - Facts:**
        - Defines compelling input as input so interesting that "you forget that it is in another language." The authors compare this to flow and being "lost in the book."
        - Argues compelling input removes the need for motivation: "you acquire whether you are interested in improving or not."
        - Documents a case ("Daniel") where a learner's language growth resumed only after he got hooked on an illustrated story series.
      - **DOK 2 - Summary & Analysis:**
        - A level-appropriate story is not enough. It has to be interesting enough that the learner forgets they are studying and wants to continue. Humor, surprise, and emotion are not decoration here. They are part of how the system gets enough reading volume.
      - **Link to source:** [Krashen, "The Compelling Input Hypothesis" (PDF)](http://www.sdkrashen.com/content/articles/the_compelling_input_hypothesis.pdf)
  - **Humor, Surprise, And Emotion**
    - **Source: Schmidt (2002), "The Humour Effect: Differential Processing and Privileged Retrieval," *Memory*.**
      - **DOK 1 - Facts:**
        - Cartoons in humorous, literal, and "weird" versions were compared. Humorous versions were remembered better, while literal and weird versions were remembered about equally. The boost came from humor, not mere bizarreness.
        - The advantage appeared only in mixed lists where humorous and non-humorous items were studied together, a von Restorff style within-list contrast.
        - Better recall of humorous items came partly at the expense of recall for the non-humorous items in the same list.
      - **DOK 2 - Summary & Analysis:**
        - Humor works best when attached to the target word or sentence, not merely placed nearby. It is also close to zero sum, so making everything funny cancels the benefit. The design rule is simple: reserve the humorous beat for the sentence carrying the new word, against an otherwise ordinary background.
      - **Link to source:** [Schmidt 2002 (PubMed)](https://pubmed.ncbi.nlm.nih.gov/11798442/)
    - **Source: Hunt (1995), "The Subtlety of Distinctiveness: What von Restorff Really Did," *Psychonomic Bulletin & Review*.**
      - **DOK 1 - Facts:**
        - The isolation (von Restorff) effect: when all but one list item are similar, the different item is remembered better; von Restorff's data showed ~.70 recall for the isolated item vs ~.40 for controls.
        - Hunt replicated the effect (n = 40) with isolated items recalled at .70 to .80.
        - Perceptual salience is *not* required. Distinctiveness comes from the item's difference from a similar context, not raw perceptual pop.
      - **DOK 2 - Summary & Analysis:**
        - A single new word among known words is an isolation setup. Distinctiveness theory predicts that the new word will be better recalled because the familiar vocabulary around it makes it stand out. This makes the one-new-word constraint about memory as well as comprehension.
      - **Link to source:** [Hunt 1995 (open PDF)](https://link.springer.com/content/pdf/10.3758/BF03214414.pdf)
  - **Attention And The Seductive-Details Risk**
    - **Source: Harp & Mayer (1998), "How Seductive Details Do Their Damage," *Journal of Educational Psychology*.**
      - **DOK 1 - Facts:**
        - Across four experiments, adding interesting-but-irrelevant "seductive details" significantly reduced recall of main ideas and problem-solving transfer.
        - Highlighting main ideas, stating objectives, or signaling structure did not remove the effect.
        - Placing seductive details at the beginning worsened the damage. Placing them at the end reduced it. The authors propose that seductive details prime the wrong schema. (Note: the "six of six / 3x recall" figure belongs to a different Harp & Mayer paper and should not be attributed here.)
      - **DOK 2 - Summary & Analysis:**
        - Fun details that are not tied to the language target can hurt learning, especially up front, and simple fixes do not rescue them. Humor, surprise, and emotion need to carry the target word rather than decorate the story around it. Off-target embellishment is a defect, not color.
      - **Link to source:** [Harp & Mayer 1998 (DOI)](https://doi.org/10.1037/0022-0663.90.3.414)
- **Fine-Tuning, Data Generation, And Evaluation**
  - **The Direct Precedent**
    - **Source: Kamzela, Lango & Dušek (2025), "SRS-Stories: Vocabulary-Constrained Multilingual Story Generation for Language Learning," EMNLP 2025 (Industry).**
      - **DOK 1 - Facts:**
        - Formalizes the task: given a known-vocabulary set and a to-learn set, generate a coherent story using only allowed words, with each new word appearing at least three times.
        - Compares generation strategies (simple prompting, planning, examples-first) and post-hoc constraint-enforcement strategies (iterative rewriting, up to five passes, with variants), all checked by an external normalize, tokenize, lemmatize validator against the allowed list.
        - Covers English (CEFR-J lists), Chinese (New HSK 3.0), and Polish (frequency-cutoff lists); primary generator was Llama 3.1 70B Instruct at temperature 0, judged by Qwen2.5 72B.
        - A constrained-beam-search baseline was the *worst* method, often ungrammatical with words out of context. Prompting plus rewriting cut out-of-vocabulary words from ~6.7% to ~0.6% while improving grammaticality and coherence.
        - Human to LLM-judge correlation was only moderate (English Pearson r about 0.46 to 0.56, lower for Chinese and Polish).
      - **DOK 2 - Summary & Analysis:**
        - SRS-Stories is the closest published precedent. It uses prompting and post-hoc rewriting on a 70B model with no fine-tuning. The open gap is whether that behavior can be distilled into a reliable small model, roughly 0.6B to 4B. Its most useful lesson is that hard constrained decoding hurt fluency, while validator-guided rewriting worked better.
      - **Link to source:** [SRS-Stories (ACL Anthology)](https://aclanthology.org/2025.emnlp-industry.44/)
  - **Why Base Models Need Specialization**
    - **Source: Zhou et al. (2023), "Instruction-Following Evaluation for Large Language Models" (IFEval).**
      - **DOK 1 - Facts:**
        - Introduces IFEval, a benchmark of prompts with verifiable instructions such as required keywords, word counts, section counts, formatting requirements, and forbidden terms.
        - Uses automatic, rule-based checking instead of subjective human or LLM-judge scoring.
        - The benchmark focuses on whether a model satisfies all explicit constraints, not whether the answer merely sounds high quality.
      - **DOK 2 - Summary & Analysis:**
        - IFEval is a good match for this project's validator problem: "use only known words," "add at most one new word per sentence," and "repeat target words" are all verifiable constraints. Hard behavioral checks matter more than fluent story quality alone.
      - **Link to source:** [IFEval (arXiv)](https://arxiv.org/abs/2311.07911)

