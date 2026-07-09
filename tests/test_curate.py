"""Second-pass curation: crafted good/bad records exercise each rejection reason."""

from islm.datagen.curate import CurationConfig, curate
from islm.vocab.lemmatize import SimpleLemmatizer

KNOWN = "the, cat, see, is, be, big, a, and, run, it"


def _record(rid, story, targets=("clue",)):
    user = f"TARGET_WORDS: {', '.join(targets)}\nKNOWN_WORDS: {KNOWN}"
    return {
        "id": rid,
        "language": "en",
        "level": "A1-A2",
        "theme": "t",
        "target_words": list(targets),
        "messages": [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": user},
            {"role": "assistant", "content": story},
        ],
    }


GOOD = (
    "The cat is big. The cat see a clue. It is a big clue. "
    "The clue is big. The cat and the clue run."
)


def _curate(records):
    # Force the deterministic English analyzer for reproducibility.
    import islm.datagen.curate as c

    c.get_analyzer = lambda lang, **kwargs: SimpleLemmatizer()
    return curate({"train": records}, CurationConfig())


def test_good_record_is_kept():
    result = _curate([_record("good", GOOD)])
    assert result.n_kept == 1
    assert not result.rejects


def test_exact_duplicate_is_rejected():
    result = _curate([_record("a", GOOD), _record("b", GOOD)])
    assert result.n_kept == 1
    assert result.rejects[0]["reject_reasons"] == ["duplicate"]


def test_too_short_is_rejected():
    result = _curate([_record("short", "The cat is big. The clue is big.")])
    assert result.n_kept == 0
    assert "too_short" in result.rejects[0]["reject_reasons"]


def test_repetitive_sentences_rejected():
    story = "\n".join(["The cat is big. The cat see a clue."] + ["The clue is big."] * 5)
    result = _curate([_record("rep", story)])
    assert "repetitive_sentences" in result.rejects[0]["reject_reasons"]


def test_out_of_vocabulary_fails_revalidation():
    # "dragon" is not in KNOWN and not a target -> OOV -> hard validation fails.
    story = "The cat is big. The cat see a clue. The clue is big. The dragon is big."
    result = _curate([_record("oov", story)])
    assert "failed_revalidation" in result.rejects[0]["reject_reasons"]


def test_report_stats():
    result = _curate([_record("a", GOOD), _record("b", GOOD)])
    assert result.reasons["duplicate"] == 1
