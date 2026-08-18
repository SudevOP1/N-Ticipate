"""
Phase 6 tests (HMM side): next_tag_distribution and
most_likely_tag_for_word, the two hooks the reranker depends on.

Run with: pytest tests/test_hmm_reranking.py -v
"""

import math

from nticipate.hmm import HMMTagger

TAGGED_CORPUS = [
    [("the", "DET"), ("dog", "NOUN"), ("runs", "VERB"), ("fast", "ADV")],
    [("the", "DET"), ("cat", "NOUN"), ("runs", "VERB")],
    [("a", "DET"), ("dog", "NOUN"), ("sleeps", "VERB"), ("quietly", "ADV")],
    [("the", "DET"), ("big", "ADJ"), ("dog", "NOUN"), ("barks", "VERB")],
    [("a", "DET"), ("cat", "NOUN"), ("sleeps", "VERB")],
]


# ---------------------------------------------------------------------------
# next_tag_distribution
# ---------------------------------------------------------------------------
def test_next_tag_distribution_sums_to_one():
    tagger = HMMTagger().fit(TAGGED_CORPUS)
    dist = tagger.next_tag_distribution(("DET",))
    assert math.isclose(sum(dist.values()), 1.0, abs_tol=1e-9)


def test_next_tag_distribution_covers_every_tag():
    tagger = HMMTagger().fit(TAGGED_CORPUS)
    dist = tagger.next_tag_distribution(("DET",))
    assert set(dist.keys()) == tagger.tags


def test_next_tag_distribution_favors_noun_after_det():
    tagger = HMMTagger().fit(TAGGED_CORPUS)
    dist = tagger.next_tag_distribution(("DET",))
    # every sentence in this corpus has NOUN (or ADJ then NOUN) after DET
    assert dist["NOUN"] > dist["VERB"]
    assert dist["NOUN"] > dist["ADV"]


def test_next_tag_distribution_empty_context_uses_initial():
    tagger = HMMTagger().fit(TAGGED_CORPUS)
    dist = tagger.next_tag_distribution(())
    assert math.isclose(sum(dist.values()), 1.0, abs_tol=1e-9)
    # every sentence in this corpus starts with DET
    assert dist["DET"] > dist["VERB"]


def test_next_tag_distribution_only_uses_most_recent_tag():
    tagger = HMMTagger().fit(TAGGED_CORPUS)
    # bigram tag model -- a longer tag_context should reduce to the same
    # result as just its last element
    dist_long = tagger.next_tag_distribution(("VERB", "DET"))
    dist_short = tagger.next_tag_distribution(("DET",))
    assert dist_long == dist_short


# ---------------------------------------------------------------------------
# most_likely_tag_for_word
# ---------------------------------------------------------------------------
def test_most_likely_tag_for_known_word():
    tagger = HMMTagger().fit(TAGGED_CORPUS)
    assert tagger.most_likely_tag_for_word("dog") == "NOUN"
    assert tagger.most_likely_tag_for_word("runs") == "VERB"


def test_most_likely_tag_falls_back_to_title_case():
    # tagger vocab has "dog" lowercase; n-gram-vocab-style lookups are
    # always lowercase already, so this mainly guards the reverse case
    # (an HMM vocab entry that happens to be capitalized)
    corpus = [[("Dog", "NOUN"), ("runs", "VERB")]]
    tagger = HMMTagger().fit(corpus)
    assert tagger.most_likely_tag_for_word("dog") == "NOUN"


def test_most_likely_tag_for_oov_word_uses_suffix_heuristic():
    tagger = HMMTagger().fit(TAGGED_CORPUS)
    assert tagger.most_likely_tag_for_word("jumping") == "VERB"  # -ing suffix
