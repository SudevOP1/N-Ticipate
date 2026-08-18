"""
Phase 1 tests. These exercise the regex-fallback path by default (no
NLTK data download required), and the real NLTK path automatically
once you've run `python setup_env.py`.

Run with: pytest tests/test_preprocess.py -v
"""

from nticipate.preprocess import (
    clean_text,
    segment_sentences,
    tokenize,
    build_vocab,
    apply_unk,
    build_truecase_map,
    apply_truecase,
    pad_sentence,
    train_dev_test_split,
    preprocess_corpus,
    UNK_TOKEN,
    START_TOKEN,
    END_TOKEN,
)


# ---------------------------------------------------------------------------
# clean_text
# ---------------------------------------------------------------------------
def test_clean_text_collapses_whitespace():
    assert clean_text("hello    world\n\n\tfoo") == "hello world foo"


def test_clean_text_normalizes_smart_quotes():
    assert clean_text("\u201cHello\u2019s world\u201d") == '"Hello\'s world"'


def test_clean_text_strips_html_tags():
    assert clean_text("<p>Hello <b>world</b></p>") == "Hello world"


def test_clean_text_handles_empty_and_none():
    assert clean_text("") == ""
    assert clean_text(None) == ""


# ---------------------------------------------------------------------------
# segment_sentences
# ---------------------------------------------------------------------------
def test_segment_sentences_english_basic():
    text = "This is one sentence. This is another one! Is this a third?"
    sentences = segment_sentences(text)
    assert len(sentences) == 3
    assert sentences[0].startswith("This is one")


def test_segment_sentences_devanagari_danda():
    # "This is Hindi. This is the second sentence."
    text = "यह हिंदी है। यह दूसरा वाक्य है।"
    sentences = segment_sentences(text)
    assert len(sentences) == 2
    assert sentences[0].endswith("।")


def test_segment_sentences_empty_text():
    assert segment_sentences("") == []


# ---------------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------------
def test_tokenize_splits_words_and_punctuation():
    tokens = tokenize("Hello, world!")
    assert "Hello" in tokens
    assert "world" in tokens
    assert "," in tokens
    assert "!" in tokens


def test_tokenize_handles_devanagari():
    tokens = tokenize("यह एक वाक्य है।")
    assert len(tokens) == 5  # 4 words + danda
    assert "।" in tokens


def test_tokenize_empty_sentence():
    assert tokenize("") == []


# ---------------------------------------------------------------------------
# vocabulary
# ---------------------------------------------------------------------------
def test_build_vocab_drops_rare_words():
    sentences = [["a", "b", "c"], ["a", "b"], ["a"]]
    # a:3, b:2, c:1
    vocab = build_vocab(sentences, min_freq=2)
    assert "a" in vocab
    assert "b" in vocab
    assert "c" not in vocab


def test_build_vocab_always_includes_specials():
    vocab = build_vocab([["a", "a"]], min_freq=1)
    assert UNK_TOKEN in vocab
    assert START_TOKEN in vocab
    assert END_TOKEN in vocab


def test_apply_unk_replaces_oov_words():
    vocab = {"a", "b", UNK_TOKEN}
    result = apply_unk(["a", "b", "c", "d"], vocab)
    assert result == ["a", "b", UNK_TOKEN, UNK_TOKEN]


def test_apply_unk_keeps_in_vocab_words_untouched():
    vocab = {"a", "b", UNK_TOKEN}
    assert apply_unk(["a", "b"], vocab) == ["a", "b"]


def test_build_vocab_is_case_insensitive_by_default():
    # "The" and "the" must count as the same word for vocab purposes,
    # per CFG['preprocessing']['lowercase_for_counts'] (default: true).
    sentences = [["The", "cat"], ["the", "cat"], ["The", "dog"]]
    vocab = build_vocab(sentences, min_freq=2)
    assert "the" in vocab
    assert "The" not in vocab  # counting key is lowercased
    assert "cat" in vocab


def test_apply_unk_lowercases_known_words():
    vocab = {"the", "cat", UNK_TOKEN}
    assert apply_unk(["The", "cat"], vocab) == ["the", "cat"]


def test_build_truecase_map_picks_most_frequent_casing():
    # "India" (capitalized, e.g. sentence-initial) appears more often
    # than "india" -- the map should prefer the majority spelling.
    sentences = [["India", "is", "large"], ["India", "is", "old"], ["india", "is", "big"]]
    tmap = build_truecase_map(sentences)
    assert tmap["india"] == "India"


def test_apply_truecase_restores_natural_casing():
    tmap = {"india": "India", "is": "is"}
    assert apply_truecase(["india", "is"], tmap) == ["India", "is"]


def test_apply_truecase_passes_through_unmapped_tokens():
    tmap = {"a": "A"}
    assert apply_truecase(["a", UNK_TOKEN], tmap) == ["A", UNK_TOKEN]


# ---------------------------------------------------------------------------
# padding
# ---------------------------------------------------------------------------
def test_pad_sentence_trigram():
    padded = pad_sentence(["the", "cat", "sat"], n=3)
    assert padded == [START_TOKEN, START_TOKEN, "the", "cat", "sat", END_TOKEN]


def test_pad_sentence_unigram_no_start_pad():
    padded = pad_sentence(["the", "cat"], n=1)
    assert padded == ["the", "cat", END_TOKEN]


def test_pad_sentence_rejects_invalid_n():
    try:
        pad_sentence(["x"], n=0)
        assert False, "expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# train/dev/test split
# ---------------------------------------------------------------------------
def test_train_dev_test_split_proportions():
    sentences = [[str(i)] for i in range(1000)]
    train, dev, test = train_dev_test_split(sentences)
    assert len(train) == 800
    assert len(dev) == 100
    assert len(test) == 100
    # no overlap, no loss
    all_ids = {t[0] for t in train} | {t[0] for t in dev} | {t[0] for t in test}
    assert len(all_ids) == 1000


def test_train_dev_test_split_is_reproducible():
    sentences = [[str(i)] for i in range(500)]
    train1, dev1, test1 = train_dev_test_split(sentences)
    train2, dev2, test2 = train_dev_test_split(sentences)
    assert train1 == train2
    assert dev1 == dev2
    assert test1 == test2


# ---------------------------------------------------------------------------
# full pipeline
# ---------------------------------------------------------------------------
def test_preprocess_corpus_end_to_end():
    docs = [
        "The cat sat on the mat. The cat also sat on the rug.",
        "A dog ran in the park.",
    ]
    unk_applied, vocab, truecase_map = preprocess_corpus(docs, min_freq=2)

    # "cat", "sat", "on", "the" all appear >= 2 times and should survive
    assert "cat" in vocab
    assert "the" in vocab
    # "dog", "ran", "park" appear once each -> should be <UNK>'d
    assert "dog" not in vocab

    flat = [tok for sent in unk_applied for tok in sent]
    assert UNK_TOKEN in flat

    # lowercase "the" (3 occurrences: "on the mat", "on the rug", "in
    # the park") outnumbers capitalized "The" (2, both sentence-initial)
    assert truecase_map["the"] == "the"
