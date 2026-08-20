"""Phase 4 tests — HMM POS tagger, Viterbi decoding and unknown-word handling."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from nticipate.hmm import (
    NUM_TAGS,
    PROPER_TAGS,
    PUNCT_TAGS,
    UNKNOWN_PRIORS,
    UNKNOWN_STRATEGIES,
    HMMTagger,
    MostFrequentTagBaseline,
    candidate_tags,
    collect_predictions,
    confusion_matrix,
    confusion_pairs,
    evaluate_tagger,
    load_tagged_sentences,
    read_conll,
    smoothing_sweep,
    train_tagger,
    train_test_split_tagged,
    unknown_word_ablation,
    without_oov,
    write_conll,
)

# A corpus small enough to reason about by hand, but with the one property
# that matters: "runs" and "walks" appear as both NOUN and VERB, so only the
# surrounding tags can disambiguate them.
TRAIN = [
    [("the", "DET"), ("dog", "NOUN"), ("runs", "VERB")],
    [("the", "DET"), ("cat", "NOUN"), ("walks", "VERB")],
    [("a", "DET"), ("dog", "NOUN"), ("walks", "VERB")],
    [("the", "DET"), ("runs", "NOUN"), ("help", "VERB")],
    [("a", "DET"), ("cat", "NOUN"), ("runs", "VERB")],
    [("dogs", "NOUN"), ("walk", "VERB")],
]

HELD_OUT = [
    [("the", "DET"), ("cat", "NOUN"), ("runs", "VERB")],
    [("a", "DET"), ("dog", "NOUN"), ("walks", "VERB")],
]


def make(**kwargs) -> HMMTagger:
    kwargs.setdefault("smoothing_k", 0.1)
    return HMMTagger(**kwargs).fit(TRAIN)


# ------------------------------------------------------------ construction

def test_rejects_zero_smoothing():
    # k = 0 is MLE, which makes every sentence with an unseen word undecodable.
    with pytest.raises(ValueError):
        HMMTagger(smoothing_k=0.0)


def test_rejects_negative_smoothing():
    with pytest.raises(ValueError):
        HMMTagger(smoothing_k=-1.0)


def test_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        HMMTagger(unknown_strategy="telepathy")


def test_rejects_unknown_prior():
    with pytest.raises(ValueError):
        HMMTagger(unknown_prior="vibes")


def test_rejects_empty_corpus():
    with pytest.raises(ValueError):
        HMMTagger().fit([])


def test_unfitted_tagger_refuses_to_decode():
    with pytest.raises(RuntimeError):
        HMMTagger().viterbi(["the", "dog"])


def test_repr_is_informative():
    text = repr(make())
    assert "tags=" in text and "vocab=" in text


def test_unfitted_repr():
    assert repr(HMMTagger()) == "HMMTagger(unfitted)"


# ------------------------------------------------------- model construction

def test_learns_the_tagset():
    assert make().tags == ["DET", "NOUN", "VERB"]


def test_vocab_size_counts_types_not_tokens():
    tagger = make()
    assert tagger.vocab_size == len({w for s in TRAIN for w, _ in s})


def test_initial_distribution_sums_to_one():
    tagger = make()
    assert np.exp(tagger.log_initial).sum() == pytest.approx(1.0)


def test_transitions_and_final_sum_to_one_per_row():
    # Every tag occurrence is followed by exactly one thing: another tag, or
    # the end of the sentence. If that does not sum to 1 the smoothing
    # denominator is wrong.
    tagger = make()
    row_totals = np.exp(tagger.log_transition).sum(axis=1) + np.exp(tagger.log_final)
    assert row_totals == pytest.approx(np.ones(len(tagger.tags)))


def test_emissions_and_unseen_class_sum_to_one_per_row():
    tagger = make()
    seen = np.exp(tagger.log_emission).sum(axis=1)
    unseen = np.exp(tagger.unk_priors["laplace"])
    assert (seen + unseen) == pytest.approx(np.ones(len(tagger.tags)))


def test_known_word_uses_the_emission_matrix():
    tagger = make()
    column = tagger.emission_column("dog")
    assert column is tagger.log_emission[:, tagger._word_index["dog"]] or \
        np.allclose(column, tagger.log_emission[:, tagger._word_index["dog"]])


def test_case_backoff_treats_capitalised_known_word_as_known():
    tagger = make()
    assert tagger.knows("The")
    assert np.allclose(tagger.emission_column("The"), tagger.emission_column("the"))


def test_unknown_word_is_not_known():
    assert not make().knows("flumping")


# ------------------------------------------------------------ decoding

def brute_force_best(tagger: HMMTagger, tokens: list[str]) -> list[str]:
    """Score every possible tag sequence. Only tractable because TRAIN is tiny."""
    best_path, best_score = None, -math.inf
    for candidate in itertools.product(tagger.tags, repeat=len(tokens)):
        score = tagger.sequence_log_probability(list(zip(tokens, candidate)))
        if score > best_score:
            best_path, best_score = list(candidate), score
    return best_path


@pytest.mark.parametrize("tokens", [
    ["the", "dog", "runs"],
    ["a", "cat", "walks"],
    ["the", "runs", "help"],
    ["dogs", "walk"],
    ["the", "flumping", "dog"],
])
def test_viterbi_matches_brute_force(tokens):
    # The real correctness check on the decoder: exhaustive enumeration of
    # every tag sequence must agree with the dynamic program.
    tagger = make()
    assert tagger.viterbi(tokens) == brute_force_best(tagger, tokens)


def test_viterbi_score_matches_its_own_path_score():
    tagger = make()
    tokens = ["the", "dog", "runs"]
    trellis = tagger.trellis(tokens)
    scored = tagger.sequence_log_probability(list(zip(tokens, trellis.path)))
    assert trellis.best_score == pytest.approx(scored)


def test_context_disambiguates_an_ambiguous_word():
    # "runs" is a NOUN after a determiner and a VERB after a noun. Nothing
    # about the word itself can tell them apart -- only the tag context can,
    # which is the entire argument for using an HMM over a lookup table.
    tagger = make()
    assert tagger.viterbi(["the", "dog", "runs"])[-1] == "VERB"
    assert tagger.viterbi(["the", "runs", "help"])[1] == "NOUN"


def test_tag_returns_word_tag_pairs():
    assert make().tag(["the", "dog"]) == [("the", "DET"), ("dog", "NOUN")]


def test_empty_input_decodes_to_empty_output():
    tagger = make()
    assert tagger.viterbi([]) == []
    assert tagger.trellis([]).path == []


def test_sequence_log_probability_of_empty_is_zero():
    assert make().sequence_log_probability([]) == 0.0


def test_sequence_log_probability_rejects_unseen_tag():
    assert make().sequence_log_probability([("the", "PARTICLE")]) < -1e29


def test_trellis_shape_and_backpointers():
    tagger = make()
    trellis = tagger.trellis(["the", "dog", "runs"])
    assert trellis.scores.shape == (len(tagger.tags), 3)
    assert (trellis.backpointers[:, 0] == -1).all()   # nothing precedes token 0
    assert (trellis.backpointers[:, 1:] >= 0).all()


def test_trellis_column_is_sorted_best_first():
    values = list(make().trellis(["the", "dog"]).column(1).values())
    assert values == sorted(values, reverse=True)


# --------------------------------------------------------- log vs linear

def test_log_and_linear_space_agree_on_short_input():
    tagger = make()
    tokens = ["the", "dog", "runs"]
    assert tagger.trellis(tokens, log_space=True).path == \
        tagger.trellis(tokens, log_space=False).path


def test_linear_space_underflows_on_long_input_and_log_space_does_not():
    # The whole reason the decoder works in log space. ~120 tokens of Treebank
    # is enough in practice; the toy model here has larger per-token
    # probabilities, so give it a longer run.
    tagger = make()
    tokens = ["the", "dog", "runs"] * 200
    assert tagger.trellis(tokens, log_space=False).underflowed
    assert not tagger.trellis(tokens, log_space=True).underflowed


def test_log_space_scores_stay_finite_on_long_input():
    scores = make().trellis(["the", "dog", "runs"] * 200, log_space=True).scores
    assert np.isfinite(scores).all()


# --------------------------------------------------- unknown-word heuristics

@pytest.mark.parametrize("word,expected", [
    ("flumping", ("VBG", "VERB")),
    ("flumped", ("VBD", "VERB")),
    ("bizarrely", ("RB", "ADV")),
    ("flumpation", ("NN", "NOUN")),
    ("flumpness", ("NN", "NOUN")),
    ("flumpous", ("JJ", "ADJ")),
])
def test_suffix_rules_fire(word, expected):
    assert candidate_tags(word) == expected


def test_longest_suffix_wins():
    # "flumpation" ends in both "tion" and "s"-less "on"; the longer rule must
    # win, otherwise the one-character rules swallow everything.
    assert candidate_tags("flumpation") == ("NN", "NOUN")


def test_short_words_do_not_trigger_suffix_rules():
    # "is" ends in "s" but is not a plural, and "ly" is not an adverb.
    assert candidate_tags("is") == ()
    assert candidate_tags("ly") == ()


def test_punctuation_and_numbers_are_recognised_by_shape():
    assert candidate_tags("!!") == PUNCT_TAGS
    assert candidate_tags("1990") == NUM_TAGS
    assert candidate_tags("3.5%") == NUM_TAGS


def test_capitalisation_means_proper_noun_only_mid_sentence():
    assert candidate_tags("Vinken", is_first=False) == PROPER_TAGS
    assert candidate_tags("Vinken", is_first=True) != PROPER_TAGS


def test_empty_word_has_no_candidates():
    assert candidate_tags("") == ()


def test_suffix_strategy_moves_mass_towards_the_guessed_tag():
    # The boost is relative: what has to grow is the gap between the guessed
    # tag and the rest, since a constant added to a whole trellis column
    # cannot change which path wins.
    suffixed = make(unknown_strategy="suffix")
    plain = make(unknown_strategy="uniform")
    verb, noun = suffixed._tag_index["VERB"], suffixed._tag_index["NOUN"]

    def gap(tagger):
        column = tagger.emission_column("flumping")
        return column[verb] - column[noun]

    assert gap(suffixed) > gap(plain)


def test_a_wrong_suffix_guess_is_survivable():
    # The heuristic keeps 1 - SUFFIX_CONFIDENCE on every other tag, so a bad
    # guess costs an order of magnitude rather than making the truth impossible.
    tagger = make(unknown_strategy="suffix")
    column = tagger.emission_column("flumping")
    assert np.isfinite(column).all()
    assert column[tagger._tag_index["NOUN"]] > -1e29


def test_unknown_strategy_without_a_matching_tag_falls_back_to_the_prior():
    # This corpus has no punctuation tag, so none of PUNCT_TAGS exists and the
    # boost must be a no-op rather than an exception or an empty distribution.
    tagger = make(unknown_strategy="suffix")
    assert not set(PUNCT_TAGS) & set(tagger.tags)
    assert np.allclose(tagger.emission_column("!!"), tagger.log_emission_unk)


@pytest.mark.parametrize("strategy", UNKNOWN_STRATEGIES)
def test_every_advertised_strategy_decodes(strategy):
    assert len(make(unknown_strategy=strategy).viterbi(["the", "flumping"])) == 2


@pytest.mark.parametrize("prior", UNKNOWN_PRIORS)
def test_every_advertised_prior_decodes(prior):
    assert len(make(unknown_prior=prior).viterbi(["the", "flumping"])) == 2


def test_prior_switch_changes_the_unseen_emission():
    tagger = make()
    assert not np.allclose(tagger.unk_priors["hapax"], tagger.unk_priors["laplace"])


def test_switching_prior_on_a_fitted_tagger_takes_effect():
    tagger = make(unknown_strategy="uniform", unknown_prior="laplace")
    before = tagger.emission_column("flumpxyz").copy()
    tagger.unknown_prior = "hapax"
    tagger.reset_cache()
    assert not np.allclose(before, tagger.emission_column("flumpxyz"))


def test_hapax_prior_falls_back_when_no_word_occurs_once():
    # Every word appears twice, so there are no hapax words to learn from.
    doubled = [[("a", "DET"), ("b", "NOUN")], [("a", "DET"), ("b", "NOUN")]]
    tagger = HMMTagger(smoothing_k=0.1).fit(doubled)
    assert np.allclose(tagger.unk_priors["hapax"], tagger.unk_priors["laplace"])


# --------------------------------------------------------------- evaluation

def test_evaluation_accounting_adds_up():
    result = make().evaluate(HELD_OUT)
    assert result.tokens == result.known_tokens + result.unknown_tokens
    assert sum(total for _, total in result.per_tag.values()) == result.tokens


def test_perfect_tagger_scores_one():
    assert make().evaluate(TRAIN).accuracy == pytest.approx(1.0)


def test_oov_rate_is_zero_when_every_word_was_seen():
    assert make().evaluate(HELD_OUT).oov_rate == 0.0


def test_oov_rate_counts_unseen_words():
    result = make().evaluate([[("the", "DET"), ("flumping", "VERB")]])
    assert result.oov_rate == pytest.approx(0.5)


def test_evaluation_of_empty_corpus_does_not_divide_by_zero():
    result = make().evaluate([])
    assert result.accuracy == 0.0
    assert result.oov_rate == 0.0
    assert result.unknown_accuracy == 0.0


def test_per_tag_accuracy_is_reported_per_gold_tag():
    accuracy = make().evaluate(HELD_OUT).tag_accuracy()
    assert set(accuracy) == {"DET", "NOUN", "VERB"}


def test_as_dict_carries_the_headline_numbers():
    keys = set(make().evaluate(HELD_OUT).as_dict())
    assert keys == {"tokens", "accuracy", "known_accuracy", "unknown_accuracy",
                    "oov_rate"}


def test_collect_predictions_returns_one_row_per_token():
    rows = collect_predictions(make(), HELD_OUT)
    assert len(rows) == sum(len(s) for s in HELD_OUT)
    assert all(len(row) == 3 for row in rows)


def test_confusion_matrix_totals_match_the_gold_counts():
    rows = collect_predictions(make(), HELD_OUT)
    tags, matrix = confusion_matrix(rows)
    assert matrix.sum() == len(rows)
    assert matrix.shape == (len(tags), len(tags))


def test_confusion_pairs_lists_only_mistakes():
    rows = [("x", "NOUN", "VERB"), ("y", "NOUN", "VERB"), ("z", "DET", "DET")]
    assert confusion_pairs(rows) == [("NOUN", "VERB", 2)]


def test_without_oov_filters_out_unseen_words():
    tagger = make()
    kept = without_oov(tagger, HELD_OUT + [[("flumping", "VERB")]])
    assert kept == HELD_OUT


# ----------------------------------------------------------------- baseline

def test_baseline_learns_the_most_frequent_tag_per_word():
    baseline = MostFrequentTagBaseline().fit(TRAIN)
    assert baseline.tag(["the", "dog"]) == [("the", "DET"), ("dog", "NOUN")]


def test_baseline_falls_back_to_the_corpus_default_tag():
    baseline = MostFrequentTagBaseline().fit(TRAIN)
    assert baseline.predict_tags(["flumping"]) == [baseline.default_tag]


def test_baseline_is_context_free_and_the_hmm_is_not():
    # "runs" is tagged VERB more often than NOUN, so the baseline must get the
    # NOUN reading wrong in every context. This is the gap the HMM exists to
    # close, stated as a test rather than as a claim in the report.
    baseline = MostFrequentTagBaseline().fit(TRAIN)
    tagger = make()
    assert baseline.predict_tags(["the", "runs", "help"])[1] == "VERB"
    assert tagger.viterbi(["the", "runs", "help"])[1] == "NOUN"


def test_evaluate_tagger_accepts_either_tagger():
    for tagger in (make(), MostFrequentTagBaseline().fit(TRAIN)):
        assert evaluate_tagger(tagger, HELD_OUT).tokens == 6


def test_baseline_repr():
    assert "MostFrequentTagBaseline" in repr(MostFrequentTagBaseline().fit(TRAIN))


# -------------------------------------------------------------- persistence

def test_save_load_round_trip_preserves_decoding(tmp_path):
    tagger = make()
    tagger.save(tmp_path / "tagger.pkl")
    restored = HMMTagger.load(tmp_path / "tagger.pkl")
    tokens = ["the", "dog", "runs", "flumping"]
    assert restored.viterbi(tokens) == tagger.viterbi(tokens)


def test_save_load_round_trip_preserves_settings(tmp_path):
    tagger = make(unknown_strategy="most_frequent_tag", unknown_prior="laplace")
    tagger.save(tmp_path / "tagger.pkl")
    restored = HMMTagger.load(tmp_path / "tagger.pkl")
    assert restored.unknown_strategy == "most_frequent_tag"
    assert restored.unknown_prior == "laplace"
    assert restored.smoothing_k == tagger.smoothing_k
    assert set(restored.unk_priors) == set(tagger.unk_priors)


# ------------------------------------------------------------ corpus i/o

def test_conll_round_trip(tmp_path):
    path = write_conll(tmp_path / "corpus.conll", TRAIN)
    assert read_conll(path) == TRAIN


def test_conll_reader_skips_comments_and_blank_lines(tmp_path):
    path = tmp_path / "corpus.conll"
    path.write_text("# a comment\nthe\tDET\ndog\tNOUN\n\n\ncat\tNOUN\n",
                    encoding="utf-8")
    assert read_conll(path) == [[("the", "DET"), ("dog", "NOUN")],
                                [("cat", "NOUN")]]


def test_conll_reader_accepts_space_separated_columns(tmp_path):
    path = tmp_path / "corpus.conll"
    path.write_text("the DET\ndog NOUN\n", encoding="utf-8")
    assert read_conll(path) == [[("the", "DET"), ("dog", "NOUN")]]


def test_conll_round_trip_survives_non_latin_text(tmp_path):
    # Phase 5 reads Hindi through this same function.
    hindi = [[("यह", "PRON"), ("किताब", "NOUN"), ("।", "PUNCT")]]
    path = write_conll(tmp_path / "hindi.conll", hindi)
    assert read_conll(path) == hindi


# -------------------------------------------------------------- splitting

def test_split_is_exhaustive_and_disjoint():
    train, test = train_test_split_tagged(TRAIN * 5, train=0.8)
    assert len(train) + len(test) == 30
    assert len(test) == 6


def test_split_is_deterministic():
    first = train_test_split_tagged(TRAIN * 5, seed=7)
    second = train_test_split_tagged(TRAIN * 5, seed=7)
    assert first == second


def test_different_seeds_shuffle_differently():
    a, _ = train_test_split_tagged(TRAIN * 20, seed=1)
    b, _ = train_test_split_tagged(TRAIN * 20, seed=2)
    assert a != b


def test_split_rejects_impossible_fractions():
    with pytest.raises(ValueError):
        train_test_split_tagged(TRAIN, train=1.5)


def test_split_does_not_alias_the_input():
    sentences = [list(s) for s in TRAIN]
    train, _ = train_test_split_tagged(sentences)
    train[0].append(("injected", "NOUN"))
    assert all(("injected", "NOUN") not in s for s in sentences)


# ------------------------------------------------------------- experiments

def test_train_tagger_accepts_explicit_sentences():
    assert train_tagger(TRAIN, smoothing_k=0.1).is_fitted


def test_ablation_covers_every_cell():
    rows = unknown_word_ablation(TRAIN, HELD_OUT)
    assert len(rows) == len(UNKNOWN_PRIORS) * len(UNKNOWN_STRATEGIES)
    assert {r["prior"] for r in rows} == set(UNKNOWN_PRIORS)


def test_ablation_leaves_known_word_accuracy_alone():
    # Both knobs only touch words outside the vocabulary, so the known-word
    # column must be identical in every row. If it moves, a knob is leaking.
    rows = unknown_word_ablation(TRAIN, HELD_OUT)
    assert len({round(r["known_accuracy"], 12) for r in rows}) == 1


def test_smoothing_sweep_reports_one_row_per_value():
    rows = smoothing_sweep(TRAIN, HELD_OUT, values=(1.0, 0.01))
    assert [r["k"] for r in rows] == [1.0, 0.01]


# ------------------------------------------------------- real corpus (slow)

def _treebank(limit: int = 1500):
    try:
        return load_tagged_sentences("english", limit=limit)
    except Exception:
        pytest.skip("tagged English corpus not available -- run setup_env.py")


def test_treebank_loads_as_word_tag_pairs():
    sentences = _treebank(limit=20)
    assert all(isinstance(w, str) and isinstance(t, str)
               for s in sentences for w, t in s)


def test_hmm_beats_the_most_frequent_tag_baseline():
    """The Phase 4 acceptance criterion, as a test rather than a claim."""
    sentences = _treebank()
    train, test = train_test_split_tagged(sentences)
    hmm = HMMTagger().fit(train).evaluate(test)
    baseline = MostFrequentTagBaseline().fit(train).evaluate(test)
    assert hmm.accuracy > baseline.accuracy


def test_hmm_handles_unseen_words_better_than_the_baseline():
    sentences = _treebank()
    train, test = train_test_split_tagged(sentences)
    hmm = HMMTagger().fit(train).evaluate(test)
    baseline = MostFrequentTagBaseline().fit(train).evaluate(test)
    assert hmm.unknown_accuracy > baseline.unknown_accuracy


def test_shipped_smoothing_beats_add_one():
    """Add-one is the textbook default and measurably the wrong one here."""
    sentences = _treebank()
    train, test = train_test_split_tagged(sentences)
    rows = {r["k"]: r["accuracy"] for r in smoothing_sweep(train, test,
                                                           values=(1.0, 0.01))}
    assert rows[0.01] > rows[1.0]
