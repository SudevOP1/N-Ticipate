"""
Phase 4 tests: HMM POS tagger (transition/emission counting, smoothing,
Viterbi decoding, unknown-word suffix heuristics, accuracy).

Uses a tiny hand-constructed tagged corpus with genuine tag ambiguity
("dog" always NOUN, but "run"/"runs" could plausibly be VERB or NOUN)
so Viterbi actually has decisions to make, not just lookups.

Run with: pytest tests/test_hmm.py -v
"""

import math

from nticipate.hmm import HMMTagger, _is_devanagari


# Small corpus: DET NOUN VERB [ADV] pattern, repeated with variation,
# plus one ADJ example -- enough structure for transitions to matter.
TAGGED_CORPUS = [
    [("the", "DET"), ("dog", "NOUN"), ("runs", "VERB"), ("fast", "ADV")],
    [("the", "DET"), ("cat", "NOUN"), ("runs", "VERB")],
    [("a", "DET"), ("dog", "NOUN"), ("sleeps", "VERB"), ("quietly", "ADV")],
    [("the", "DET"), ("big", "ADJ"), ("dog", "NOUN"), ("barks", "VERB")],
    [("a", "DET"), ("cat", "NOUN"), ("sleeps", "VERB")],
]


# ---------------------------------------------------------------------------
# fit() / counting
# ---------------------------------------------------------------------------
def test_fit_learns_correct_tag_set():
    tagger = HMMTagger().fit(TAGGED_CORPUS)
    assert tagger.tags == {"DET", "NOUN", "VERB", "ADV", "ADJ"}


def test_fit_counts_emissions_correctly():
    tagger = HMMTagger().fit(TAGGED_CORPUS)
    assert tagger.emission["NOUN"]["dog"] == 3
    assert tagger.emission["NOUN"]["cat"] == 2
    assert tagger.emission["VERB"]["runs"] == 2


def test_fit_counts_initial_tags_correctly():
    tagger = HMMTagger().fit(TAGGED_CORPUS)
    # every sentence in this corpus starts with DET
    assert tagger.initial["DET"] == 5


def test_fit_counts_transitions_correctly():
    tagger = HMMTagger().fit(TAGGED_CORPUS)
    # DET -> NOUN happens in every sentence except the one with ADJ in between
    assert tagger.transition["DET"]["NOUN"] == 4
    assert tagger.transition["DET"]["ADJ"] == 1
    assert tagger.transition["ADJ"]["NOUN"] == 1


# ---------------------------------------------------------------------------
# smoothed log-probabilities
# ---------------------------------------------------------------------------
def test_log_emission_known_word_is_valid_log_prob():
    tagger = HMMTagger().fit(TAGGED_CORPUS)
    log_p = tagger._log_emission("NOUN", "dog")
    assert log_p <= 0.0  # log of a probability <= 1
    assert math.exp(log_p) > 0.0


def test_log_transition_never_exactly_zero_probability():
    tagger = HMMTagger().fit(TAGGED_CORPUS)
    # ADV -> DET never observed in this corpus, but Laplace smoothing
    # should still give it a small non-zero (finite log) probability
    log_p = tagger._log_transition("ADV", "DET")
    assert log_p > float("-inf")


def test_log_initial_favors_observed_initial_tag():
    tagger = HMMTagger().fit(TAGGED_CORPUS)
    assert tagger._log_initial("DET") > tagger._log_initial("VERB")


# ---------------------------------------------------------------------------
# unknown-word suffix heuristics
# ---------------------------------------------------------------------------
def test_suffix_heuristic_ing_and_ed_guess_verb():
    tagger = HMMTagger().fit(TAGGED_CORPUS)
    assert tagger._guess_tag_by_suffix("jumping") == "VERB"
    assert tagger._guess_tag_by_suffix("jumped") == "VERB"


def test_suffix_heuristic_ly_guesses_adv():
    tagger = HMMTagger().fit(TAGGED_CORPUS)
    assert tagger._guess_tag_by_suffix("quickly") == "ADV"


def test_suffix_heuristic_capitalized_word_guesses_noun_when_no_nnp():
    tagger = HMMTagger().fit(TAGGED_CORPUS)  # tagset has no NNP
    assert tagger._guess_tag_by_suffix("London") == "NOUN"


def test_suffix_heuristic_routes_devanagari_words_separately():
    tagger = HMMTagger().fit(TAGGED_CORPUS)
    # "khelना" style ending in ना -- verb infinitive marker
    guessed = tagger._guess_tag_by_suffix("खेलना")
    assert guessed == "VERB"


def test_is_devanagari_detection():
    assert _is_devanagari("राम") is True
    assert _is_devanagari("hello") is False
    assert _is_devanagari("") is False


def test_log_emission_unknown_word_prefers_guessed_tag():
    tagger = HMMTagger().fit(TAGGED_CORPUS)
    # "jumping" is unseen -> suffix heuristic guesses VERB
    log_p_verb = tagger._log_emission("VERB", "jumping")
    log_p_noun = tagger._log_emission("NOUN", "jumping")
    assert log_p_verb > log_p_noun


# ---------------------------------------------------------------------------
# Viterbi decoding
# ---------------------------------------------------------------------------
def test_viterbi_empty_input_returns_empty():
    tagger = HMMTagger().fit(TAGGED_CORPUS)
    assert tagger.viterbi([]) == []


def test_viterbi_returns_one_tag_per_token():
    tagger = HMMTagger().fit(TAGGED_CORPUS)
    tokens = ["the", "dog", "runs"]
    tags = tagger.viterbi(tokens)
    assert len(tags) == len(tokens)
    assert all(t in tagger.tags for t in tags)


def test_viterbi_correctly_tags_seen_sentence():
    tagger = HMMTagger().fit(TAGGED_CORPUS)
    tokens = ["the", "dog", "runs", "fast"]
    assert tagger.viterbi(tokens) == ["DET", "NOUN", "VERB", "ADV"]


def test_viterbi_handles_unknown_word_via_suffix_heuristic():
    tagger = HMMTagger().fit(TAGGED_CORPUS)
    # "jumping" never seen in training; DET context should still push
    # toward VERB given the suffix guess + transition pattern
    tokens = ["the", "dog", "jumping"]
    tags = tagger.viterbi(tokens)
    assert tags[0] == "DET"
    assert tags[1] == "NOUN"


# ---------------------------------------------------------------------------
# accuracy / evaluation
# ---------------------------------------------------------------------------
def test_accuracy_is_perfect_on_a_trivial_unambiguous_corpus():
    corpus = [[("a", "DET"), ("b", "NOUN")], [("a", "DET"), ("b", "NOUN")]]
    tagger = HMMTagger().fit(corpus)
    assert tagger.accuracy(corpus) == 1.0


def test_accuracy_returns_zero_for_empty_input():
    tagger = HMMTagger().fit(TAGGED_CORPUS)
    assert tagger.accuracy([]) == 0.0


def test_accuracy_beats_most_common_tag_baseline():
    tagger = HMMTagger().fit(TAGGED_CORPUS)
    acc = tagger.accuracy(TAGGED_CORPUS)
    most_common = tagger._most_common_tag()
    total = sum(len(s) for s in TAGGED_CORPUS)
    baseline_correct = sum(1 for s in TAGGED_CORPUS for _, t in s if t == most_common)
    baseline_acc = baseline_correct / total
    assert acc > baseline_acc


def test_confusion_pairs_sums_to_total_tokens():
    tagger = HMMTagger().fit(TAGGED_CORPUS)
    pairs = tagger.confusion_pairs(TAGGED_CORPUS)
    total_tokens = sum(len(s) for s in TAGGED_CORPUS)
    assert sum(pairs.values()) == total_tokens


# ---------------------------------------------------------------------------
# next_tag_distribution (implemented in Phase 6 -- see test_hmm_reranking.py
# for full coverage; this just confirms it no longer raises)
# ---------------------------------------------------------------------------
def test_next_tag_distribution_no_longer_raises():
    tagger = HMMTagger().fit(TAGGED_CORPUS)
    dist = tagger.next_tag_distribution(("DET",))
    assert isinstance(dist, dict)
    assert set(dist.keys()) == tagger.tags
