from islm.vocab.tokenize import split_sentences, word_surfaces


def test_split_sentences_on_punctuation_and_newlines():
    text = "The cat ran. Was it fast? Yes!\nA new line here."
    assert split_sentences(text) == [
        "The cat ran.",
        "Was it fast?",
        "Yes!",
        "A new line here.",
    ]


def test_word_surfaces_drops_punctuation_and_numbers():
    assert word_surfaces("The cat, 42 dogs; don't stop!") == [
        "The",
        "cat",
        "dogs",
        "don't",
        "stop",
    ]


def test_split_sentences_empty():
    assert split_sentences("   ") == []
