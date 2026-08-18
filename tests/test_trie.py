"""
Phase 3 tests: prefix trie.

Run with: pytest tests/test_trie.py -v
"""

from nticipate.trie import Trie
from nticipate.preprocess import START_TOKEN, END_TOKEN, UNK_TOKEN


def test_insert_and_contains():
    t = Trie()
    t.insert("cat")
    assert "cat" in t
    assert "ca" not in t  # prefix, not itself a full word
    assert "dog" not in t


def test_words_with_prefix_basic():
    t = Trie()
    for w in ["cat", "car", "cart", "dog"]:
        t.insert(w)
    results = t.words_with_prefix("ca")
    assert set(results) == {"cat", "car", "cart"}


def test_words_with_prefix_no_match_returns_empty():
    t = Trie()
    t.insert("cat")
    assert t.words_with_prefix("zzz") == []


def test_words_with_prefix_empty_prefix_returns_everything():
    t = Trie()
    for w in ["cat", "dog"]:
        t.insert(w)
    assert set(t.words_with_prefix("")) == {"cat", "dog"}


def test_words_with_prefix_respects_limit():
    t = Trie()
    for w in ["cat", "car", "cart", "cart2", "cartography"]:
        t.insert(w)
    results = t.words_with_prefix("ca", limit=2)
    assert len(results) == 2


def test_words_with_prefix_is_alphabetically_sorted():
    t = Trie()
    for w in ["cart", "car", "cat"]:
        t.insert(w)
    assert t.words_with_prefix("ca") == ["car", "cart", "cat"]


def test_prefix_that_is_itself_a_word_is_included():
    t = Trie()
    t.insert("car")
    t.insert("cart")
    results = t.words_with_prefix("car")
    assert "car" in results
    assert "cart" in results


def test_from_vocab_excludes_special_tokens_by_default():
    vocab = {"cat", "dog", START_TOKEN, END_TOKEN, UNK_TOKEN}
    t = Trie.from_vocab(vocab)
    assert "cat" in t
    assert START_TOKEN not in t
    assert END_TOKEN not in t
    assert UNK_TOKEN not in t


def test_from_vocab_custom_exclude_set():
    vocab = {"cat", "dog", "the"}
    t = Trie.from_vocab(vocab, exclude={"the"})
    assert "cat" in t
    assert "dog" in t
    assert "the" not in t


def test_devanagari_words():
    t = Trie.from_vocab({"राम", "राधा", "सीता"})
    results = t.words_with_prefix("रा")
    assert set(results) == {"राम", "राधा"}
