"""Phase 1 tests — preprocessing pipeline."""

from __future__ import annotations

import pytest

from nticipate.preprocess import (
    Corpus,
    Vocab,
    apply_truecase,
    apply_unk,
    build_truecase_map,
    build_vocab,
    clean_text,
    corpus_stats,
    coverage_curve,
    frequency_table,
    load_corpus_sentences,
    oov_rate,
    pad_sentence,
    preprocess_corpus,
    preprocess_sentences,
    segment_sentences,
    tokenize,
    tokenize_text,
    train_dev_test_split,
    truecase_word,
    zipf_points,
)

SAMPLE = (
    "The quick brown fox jumps over the lazy dog. "
    "It was the best of times, it was the worst of times. "
    "India is a large country. India has many languages."
)

HINDI = "यह एक वाक्य है। भारत एक बड़ा देश है। मुझे हिंदी पसंद है।"


# ------------------------------------------------------------- clean_text

def test_clean_text_squeezes_whitespace():
    assert clean_text("  a   b \n\t c  ") == "a b c"


def test_clean_text_strips_urls():
    out = clean_text("see https://example.com/x for more")
    assert "http" not in out and "example.com" not in out
    assert "see" in out and "for more" in out


def test_clean_text_strips_emails():
    assert "@" not in clean_text("mail me at a.b@test.co.in now")


def test_clean_text_normalizes_curly_quotes():
    assert clean_text("don’t") == "don't"


def test_clean_text_nfc_normalizes_devanagari():
    # U+0928 U+093C (na + nukta) is the decomposed form of U+0929.
    decomposed = "ऩ"
    assert clean_text(decomposed) == "ऩ"


def test_clean_text_keeps_punctuation():
    # Punctuation survives cleaning; it is a real event for the LM.
    assert clean_text("Hello, world!") == "Hello, world!"


def test_clean_text_empty():
    assert clean_text("   ") == ""


# ------------------------------------------------------ segment_sentences

def test_segment_sentences_basic():
    assert len(segment_sentences("One. Two. Three.")) == 3


def test_segment_sentences_question_and_exclamation():
    assert len(segment_sentences("Really? Yes! Fine.")) == 3


def test_segment_sentences_devanagari_danda():
    # Punkt has no Hindi model, so this must take the regex path.
    assert len(segment_sentences(HINDI)) == 3


def test_segment_sentences_empty():
    assert segment_sentences("") == []


def test_segment_regex_splitter_explicit():
    assert len(segment_sentences("A one. B two.", splitter="regex")) == 2


# --------------------------------------------------------------- tokenize

def test_tokenize_keeps_punctuation():
    # Stripping punctuation is right for classification, wrong for an LM.
    assert "," in tokenize("Hello, world")


def test_tokenize_keeps_stopwords():
    tokens = [t.lower() for t in tokenize("the cat is on the mat")]
    assert "the" in tokens and "is" in tokens and "on" in tokens


def test_tokenize_splits_contractions():
    tokens = [t.lower() for t in tokenize("I don't know", tokenizer="regex")]
    assert "do" in tokens and "n't" in tokens


def test_tokenize_devanagari():
    tokens = tokenize("भारत एक बड़ा देश है।")
    assert "भारत" in tokens
    assert "।" in tokens  # danda is its own token, not glued to "है"
    assert "है।" not in tokens


def test_tokenize_empty():
    assert tokenize("   ") == []


def test_tokenize_numbers():
    assert "3.14" in tokenize("pi is 3.14 roughly", tokenizer="regex")


def test_tokenize_text_pipeline():
    sentences = tokenize_text(SAMPLE)
    assert len(sentences) == 4
    assert all(isinstance(s, list) and s for s in sentences)


# ------------------------------------------------------------ build_vocab

def test_build_vocab_applies_min_freq():
    sentences = [["a", "a", "b"], ["a", "c"]]
    vocab = build_vocab(sentences, min_freq=2)
    assert "a" in vocab
    assert "b" not in vocab and "c" not in vocab


def test_build_vocab_is_case_insensitive():
    vocab = build_vocab([["The", "the", "THE"]], min_freq=3)
    assert "the" in vocab
    assert "The" not in vocab  # folded, not stored twice


def test_build_vocab_respects_max_size():
    sentences = [["a"] * 5 + ["b"] * 4 + ["c"] * 3]
    vocab = build_vocab(sentences, min_freq=1, max_size=2)
    assert len(vocab) == 2
    assert "a" in vocab and "b" in vocab and "c" not in vocab


def test_vocab_len_and_contains():
    vocab = build_vocab([["x", "x"]], min_freq=1)
    assert len(vocab) == 1 and "x" in vocab and "y" not in vocab


def test_vocab_round_trips_through_dict():
    vocab = build_vocab([["a", "a", "b"]], min_freq=1)
    restored = Vocab.from_dict(vocab.to_dict())
    assert restored.tokens == vocab.tokens
    assert restored.counts == vocab.counts
    assert restored.specials == vocab.specials


# -------------------------------------------------------------- apply_unk

def test_apply_unk_replaces_oov():
    vocab = build_vocab([["a", "a"]], min_freq=1)
    assert apply_unk(["a", "zzz"], vocab) == ["a", vocab.unk]


def test_apply_unk_lowercases_in_vocab_tokens():
    vocab = build_vocab([["cat", "cat"]], min_freq=1)
    assert apply_unk(["Cat"], vocab) == ["cat"]


def test_unk_is_why_perplexity_stays_finite():
    # Without an <UNK> class an unseen word has probability zero, which makes
    # held-out perplexity infinite in Phase 2.
    vocab = build_vocab([["seen", "seen"]], min_freq=2)
    assert all(t in vocab.tokens or t == vocab.unk
               for t in apply_unk(["seen", "unseen"], vocab))


# ------------------------------------------------------------- truecasing

def test_truecase_map_prefers_dominant_form():
    sentences = [["x", "India", "y"], ["z", "India", "w"], ["q", "india", "e"]]
    assert build_truecase_map(sentences)["india"] == "India"


def test_truecase_map_ignores_sentence_initial_capital():
    # "The" is capitalised only by sentence-position convention, so it must
    # not out-vote the lowercase evidence from mid-sentence occurrences.
    sentences = [["The", "cat", "sat"], ["The", "dog", "ate", "the", "bone"],
                 ["A", "man", "saw", "the", "sky"]]
    assert build_truecase_map(sentences)["the"] == "the"


def test_truecase_map_has_no_entry_for_only_initial_words():
    assert "hello" not in build_truecase_map([["Hello", "world"]])


def test_apply_truecase_restores_casing():
    truecase = {"india": "India", "is": "is", "big": "big"}
    assert apply_truecase(["india", "is", "big"], truecase) == ["India", "is", "big"]


def test_apply_truecase_capitalize_first():
    assert apply_truecase(["the", "end"], {}, capitalize_first=True) == ["The", "end"]


def test_apply_truecase_leaves_unknown_lowercase():
    assert apply_truecase(["mystery"], {}) == ["mystery"]


def test_truecase_word_single_suggestion():
    assert truecase_word("india", {"india": "India"}) == "India"
    assert truecase_word("nothing", {}) == "nothing"


def test_truecase_round_trip_reproduces_original_casing():
    # Phase 1 "done when": tokenize -> apply_truecase reproduces the original.
    text = ("Alice met Bob in Paris. "
            "Later Alice saw Bob again in Paris. "
            "Everyone likes Paris and Alice and Bob.")
    sentences = tokenize_text(text)
    truecase = build_truecase_map(sentences)
    for sentence in sentences:
        restored = apply_truecase(sentence, truecase, capitalize_first=True)
        assert restored == sentence


# ----------------------------------------------------------- pad_sentence

def test_pad_sentence_trigram_gets_two_bos():
    padded = pad_sentence(["a", "b"], n=3, bos="<s>", eos="</s>")
    assert padded == ["<s>", "<s>", "a", "b", "</s>"]


def test_pad_sentence_bigram_gets_one_bos():
    assert pad_sentence(["a"], n=2, bos="<s>", eos="</s>") == ["<s>", "a", "</s>"]


def test_pad_sentence_unigram_gets_no_bos():
    assert pad_sentence(["a"], n=1, bos="<s>", eos="</s>") == ["a", "</s>"]


def test_pad_sentence_rejects_bad_n():
    with pytest.raises(ValueError):
        pad_sentence(["a"], n=0)


def test_pad_sentence_does_not_mutate_input():
    original = ["a", "b"]
    pad_sentence(original, n=3)
    assert original == ["a", "b"]


# ------------------------------------------------------ train_dev_test_split

def test_split_partitions_without_loss():
    sentences = [[str(i)] for i in range(100)]
    train, dev, test = train_dev_test_split(sentences, 0.8, 0.1, 0.1, seed=1)
    assert len(train) == 80 and len(dev) == 10 and len(test) == 10
    joined = [s[0] for s in train + dev + test]
    assert sorted(joined, key=int) == [str(i) for i in range(100)]


def test_split_is_deterministic_for_a_seed():
    sentences = [[str(i)] for i in range(50)]
    assert train_dev_test_split(sentences, seed=7) == train_dev_test_split(sentences, seed=7)


def test_split_differs_across_seeds():
    sentences = [[str(i)] for i in range(50)]
    a, _, _ = train_dev_test_split(sentences, seed=1)
    b, _, _ = train_dev_test_split(sentences, seed=2)
    assert a != b


def test_split_rejects_fractions_not_summing_to_one():
    with pytest.raises(ValueError):
        train_dev_test_split([["a"]], 0.5, 0.3, 0.1)


def test_split_does_not_mutate_input_order():
    sentences = [[str(i)] for i in range(20)]
    train_dev_test_split(sentences, seed=3)
    assert sentences[0] == ["0"]


# ------------------------------------------------------- full pipeline

def test_preprocess_corpus_produces_all_parts():
    corpus = preprocess_corpus(SAMPLE * 5)
    assert corpus.train and corpus.vocab.tokens
    assert isinstance(corpus.truecase, dict)


def test_preprocess_corpus_vocab_built_from_train_only():
    # Building the vocab from everything would leak test vocabulary into the
    # model and flatter the Phase 2 perplexity numbers.
    corpus = preprocess_corpus(SAMPLE * 10)
    train_types = {t for s in corpus.train for t in s}
    assert corpus.vocab.tokens <= train_types | set(corpus.vocab.specials)


def test_preprocess_corpus_unks_rare_words():
    text = "cat cat cat dog dog dog. cat cat dog dog. zzz cat dog cat dog."
    corpus = preprocess_corpus(text, min_sentence_tokens=1)
    assert "zzz" not in corpus.vocab.tokens


def test_corpus_save_load_round_trip(tmp_path):
    corpus = preprocess_corpus(SAMPLE * 5)
    path = corpus.save(tmp_path / "corpus.json")
    restored = Corpus.load(path)
    assert restored.train == corpus.train
    assert restored.test == corpus.test
    assert restored.vocab.tokens == corpus.vocab.tokens
    assert restored.truecase == corpus.truecase


def test_corpus_save_preserves_devanagari(tmp_path):
    corpus = preprocess_corpus(HINDI * 5, min_sentence_tokens=1)
    restored = Corpus.load(corpus.save(tmp_path / "hi.json"))
    assert restored.train == corpus.train


def test_all_sentences_is_the_union():
    corpus = preprocess_corpus(SAMPLE * 5)
    assert len(corpus.all_sentences) == \
        len(corpus.train) + len(corpus.dev) + len(corpus.test)


# ------------------------------------------------------------- statistics

def test_corpus_stats_counts():
    stats = corpus_stats([["a", "b"], ["a", "c", "d"]])
    assert stats["sentences"] == 2
    assert stats["tokens"] == 5
    assert stats["types"] == 4
    assert stats["type_token_ratio"] == pytest.approx(0.8)
    assert stats["hapax"] == 3


def test_corpus_stats_empty():
    assert corpus_stats([])["tokens"] == 0


def test_frequency_table_is_ordered():
    table = frequency_table([["a", "a", "a", "b", "b", "c"]], top=2)
    assert table == [("a", 3), ("b", 2)]


def test_zipf_points_are_monotonically_decreasing():
    ranks, freqs = zipf_points([["a"] * 5 + ["b"] * 3 + ["c"]])
    assert ranks == [1, 2, 3]
    assert freqs == sorted(freqs, reverse=True)


def test_coverage_curve_reaches_full_coverage():
    curve = coverage_curve([["a"] * 10 + ["b"] * 5], steps=[1, 100])
    assert curve[-1][1] == pytest.approx(1.0)


def test_coverage_curve_empty():
    assert coverage_curve([]) == []


def test_oov_rate():
    vocab = build_vocab([["a", "a"]], min_freq=1)
    assert oov_rate([["a", "b"]], vocab) == pytest.approx(0.5)
    assert oov_rate([], vocab) == 0.0


# ---------------------------------------------------------- corpus loading

def test_preprocess_sentences_accepts_pretokenized():
    sentences = [["the", "cat", "sat"], ["the", "dog", "ran"]] * 20
    corpus = preprocess_sentences(sentences)
    assert corpus.train and "the" in corpus.vocab.tokens


def test_load_corpus_sentences_from_file(tmp_path):
    path = tmp_path / "mini.txt"
    path.write_text(SAMPLE, encoding="utf-8")
    sentences = load_corpus_sentences(str(path))
    assert len(sentences) == 4
    assert sentences[0][0] == "The"


def test_load_corpus_sentences_respects_limit(tmp_path):
    path = tmp_path / "mini.txt"
    path.write_text(SAMPLE, encoding="utf-8")
    assert len(load_corpus_sentences(str(path), limit=2)) == 2


def test_load_corpus_sentences_missing_raises():
    with pytest.raises(FileNotFoundError):
        load_corpus_sentences("no_such_corpus_xyz")


@pytest.mark.parametrize("name", ["brown"])
def test_nltk_corpus_is_untagged(name):
    # brown.raw() is POS-tagged ("The/at Fulton/np-tl"); .sents() is not.
    # Training on the raw form would build a vocabulary of tag-suffixed
    # pseudo-words, so this guards the loader against regressing to .raw().
    pytest.importorskip("nltk")
    try:
        sentences = load_corpus_sentences(name, limit=50)
    except FileNotFoundError:
        pytest.skip(f"{name} corpus not downloaded")
    assert not any("/" in token for sent in sentences for token in sent)
