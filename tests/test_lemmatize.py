import pytest

from islm.vocab.lemmatize import SimpleLemmatizer


@pytest.fixture
def lem():
    return SimpleLemmatizer()


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("cats", "cat"),
        ("boxes", "box"),
        ("studies", "study"),
        ("walked", "walk"),
        ("running", "run"),
        ("happier", "happy"),
        ("mice", "mouse"),  # irregular table
        ("went", "go"),  # irregular table
        ("making", "make"),  # irregular table
        ("glass", "glass"),  # no over-stripping of -ss
        ("cat", "cat"),  # base form unchanged
    ],
)
def test_simple_lemma(lem, word, expected):
    assert lem.analyze(word)[0].lemma == expected


def test_proper_noun_detection_midsentence(lem):
    tokens = lem.analyze("the cat met Milo")
    milo = tokens[-1]
    assert milo.surface == "Milo"
    assert milo.is_proper is True
    # A sentence-initial capital is treated as a normal word, not a proper noun.
    assert lem.analyze("The cat")[0].is_proper is False
