"""Multilingual support: CJK tokenization, tier loading, sampler, and end-to-end per language.

Analyzer tests that need jieba / fugashi skip cleanly when those packages are absent.
"""

import pytest

from islm.datagen.generate import make_example
from islm.datagen.scenarios import sample_scenarios
from islm.llm.client import MockLLM
from islm.vocab.languages import SUPPORTED_LANGUAGES, get_language
from islm.vocab.lemmatize import GenericAnalyzer, get_analyzer
from islm.vocab.tokenize import split_sentences
from islm.vocab.wordlists import load_advanced, load_baseline


def test_cjk_sentence_split():
    assert split_sentences("你好。世界。") == ["你好。", "世界。"]
    assert split_sentences("猫が好き。犬も好き。") == ["猫が好き。", "犬も好き。"]


def test_generic_analyzer_any_language():
    lemmas = [t.lemma for t in GenericAnalyzer().analyze("Hola Mundo dos")]
    assert lemmas == ["hola", "mundo", "dos"]


def test_supported_languages():
    assert SUPPORTED_LANGUAGES == ("en", "zh", "ja")
    assert get_language("fr").analyzer == "generic"  # graceful fallback


@pytest.mark.parametrize(("lang", "known_word"), [("en", "cat"), ("zh", "猫"), ("ja", "猫")])
def test_tiers_load(lang, known_word):
    baseline, advanced = load_baseline(lang), load_advanced(lang)
    assert len(baseline) > 20
    assert len(advanced) > 10
    assert not (baseline.lemmas & advanced.lemmas)  # disjoint tiers
    assert known_word in baseline


@pytest.mark.parametrize("lang", ["en", "zh", "ja"])
def test_sampler_multilingual(lang):
    scenarios = sample_scenarios(3, language=lang, seed=1)
    known = load_baseline(lang).lemmas
    for s in scenarios:
        assert s.language == lang
        assert s.known
        assert s.target_words
        for t in s.target_set():
            assert t not in known  # targets are genuinely new


def test_chinese_analyzer():
    pytest.importorskip("jieba")
    surfaces = [t.surface for t in get_analyzer("zh").analyze("我喜欢猫")]
    assert "猫" in surfaces and "喜欢" in surfaces


def test_japanese_analyzer_lemmatizes():
    pytest.importorskip("fugashi")
    tokens = get_analyzer("ja").analyze("魚を食べた")
    # Conjugated 食べた -> dictionary form 食べる.
    assert "食べる" in [t.lemma for t in tokens]
    assert "魚" in [t.surface for t in tokens]


@pytest.mark.parametrize(("lang", "pkg"), [("zh", "jieba"), ("ja", "fugashi")])
def test_mock_pipeline_per_language(lang, pkg):
    pytest.importorskip(pkg)
    analyzer = get_analyzer(lang)
    for scenario in sample_scenarios(3, language=lang, seed=2):
        ex = make_example(scenario, MockLLM(), lemmatizer=analyzer)
        assert ex.report.hard_pass, ex.report.failures()
        for word in scenario.target_set():
            assert ex.report.recurrence.counts[word] >= 3
