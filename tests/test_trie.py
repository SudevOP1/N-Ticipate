"""Phase 3 tests — prefix trie."""

from __future__ import annotations

from collections import Counter

import pytest

from nticipate.trie import Trie

WORDS = ["recommend", "receive", "recent", "record", "read", "cat", "car"]


def make(words=WORDS) -> Trie:
    return Trie(words)


# ----------------------------------------------------------------- inserting

def test_insert_and_contains():
    trie = make()
    assert "recent" in trie
    assert "rec" not in trie  # a prefix is not a word unless inserted as one


def test_len_counts_distinct_words():
    assert len(make()) == len(WORDS)


def test_repeated_insert_accumulates_count_not_size():
    trie = Trie()
    trie.insert("cat")
    trie.insert("cat", 4)
    assert len(trie) == 1
    assert trie.count_of("cat") == 5


def test_empty_word_is_ignored():
    trie = Trie()
    trie.insert("")
    assert len(trie) == 0


def test_count_of_unknown_word_is_zero():
    assert make().count_of("zebra") == 0


def test_from_counts():
    trie = Trie.from_counts(Counter({"cat": 3, "car": 1}))
    assert trie.count_of("cat") == 3 and len(trie) == 2


def test_has_prefix():
    trie = make()
    assert trie.has_prefix("rec")
    assert not trie.has_prefix("zzz")


# ---------------------------------------------------------------- completion

def test_complete_returns_only_matching_words():
    words = [w for w, _ in make().complete("rec", k=10)]
    assert set(words) == {"recommend", "receive", "recent", "record"}


def test_complete_ranks_by_count():
    trie = Trie()
    trie.insert("recent", 10)
    trie.insert("receive", 5)
    trie.insert("record", 1)
    assert [w for w, _ in trie.complete("rec", k=3)] == ["recent", "receive", "record"]


def test_complete_respects_k():
    assert len(make().complete("rec", k=2)) == 2


def test_complete_includes_the_prefix_itself_when_it_is_a_word():
    # Typing "cat" should still offer "cat" -- the user may be done, and
    # offering only "cats" would be worse than useless.
    trie = Trie(["cat", "cats", "catalogue"])
    assert "cat" in [w for w, _ in trie.complete("cat", k=5)]


def test_complete_on_unknown_prefix_is_empty():
    assert make().complete("zzz", k=5) == []


def test_complete_on_empty_prefix_returns_everything():
    assert len(make().complete("", k=100)) == len(WORDS)


def test_complete_honours_exclude():
    words = [w for w, _ in make().complete("rec", k=10, exclude={"recent"})]
    assert "recent" not in words


def test_complete_is_deterministic_on_ties():
    trie = Trie()
    for word in ["reb", "rea", "rec"]:
        trie.insert(word, 5)
    assert [w for w, _ in trie.complete("re", k=3)] == ["rea", "reb", "rec"]


def test_words_with_prefix_yields_counts():
    trie = Trie()
    trie.insert("cat", 7)
    assert dict(trie.words_with_prefix("ca")) == {"cat": 7}


# ------------------------------------------------------------------ metrics

def test_node_count_is_at_least_the_longest_word():
    trie = Trie(["recommend"])
    assert trie.node_count() >= len("recommend") + 1  # +1 for the root


def test_shared_prefixes_share_nodes():
    separate = Trie(["cat"]).node_count() + Trie(["car"]).node_count()
    shared = Trie(["cat", "car"]).node_count()
    assert shared < separate


def test_repr_reports_size():
    text = repr(make())
    assert "words=" in text and "nodes=" in text


# ------------------------------------------------------------------ unicode

def test_devanagari_completion():
    # The trie is character-based, so Phase 5's Hindi works with no changes.
    trie = Trie(["भारत", "भारतीय", "भाषा"])
    assert set(w for w, _ in trie.complete("भार", k=5)) == {"भारत", "भारतीय"}


@pytest.mark.parametrize("prefix,expected", [("c", 2), ("ca", 2), ("cat", 1), ("r", 5)])
def test_prefix_subtree_sizes(prefix, expected):
    assert len(make().complete(prefix, k=100)) == expected
