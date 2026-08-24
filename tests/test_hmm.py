"""Phase 4-5 tests — HMM POS tagger, Viterbi decoding, unknown words, Hindi.

Phase 5 adds no class and no decoder, so its tests live here: the code under
test is the same code, exercised on Devanagari.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from nticipate.preprocess import contains_devanagari
from nticipate.hmm import (
    DANDA,
    DEVANAGARI_SUFFIX_RULES,
    NUM_TAGS,
    PROPER_TAGS,
    PUNCT_TAGS,
    UNKNOWN_PRIORS,
    UNKNOWN_STRATEGIES,
    HMMTagger,
    MostFrequentTagBaseline,
    candidate_tags,
    clean_tagged_sentences,
    collect_predictions,
    confusion_matrix,
    confusion_pairs,
    evaluate_tagger,
    language_comparison,
    load_tagged_sentences,
    read_conll,
    smoothing_sweep,
    suffix_rule_report,
    tagged_corpus_stats,
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


# ==========================================================================
# Phase 5 — the same class, in Devanagari
# ==========================================================================

# Bureau of Indian Standards tags, as NLTK's hindi.pos uses them: NN common
# noun, NNP proper noun, VFM finite main verb, VAUX auxiliary, PREP
# postposition, PRP pronoun, JJ adjective, PUNC punctuation.
HINDI = [
    [("राम", "NNP"), ("ने", "PREP"), ("किताब", "NN"), ("पढ़ी", "VFM"), ("।", "PUNC")],
    [("सीता", "NNP"), ("ने", "PREP"), ("किताब", "NN"), ("लिखी", "VFM"), ("।", "PUNC")],
    [("यह", "PRP"), ("अच्छी", "JJ"), ("किताब", "NN"), ("है", "VAUX"), ("।", "PUNC")],
    [("राम", "NNP"), ("घर", "NN"), ("जाता", "VFM"), ("है", "VAUX"), ("।", "PUNC")],
]


def hindi_tagger(**kwargs) -> HMMTagger:
    kwargs.setdefault("smoothing_k", 0.1)
    return HMMTagger(**kwargs).fit(HINDI)


# ------------------------------------------------- Devanagari suffix rules

@pytest.mark.parametrize("word, expected", [
    ("लोगों", "NN"),          # oblique plural, the most reliable rule of the set
    ("बच्चियों", "NN"),
    ("करने", "VNN"),          # oblique infinitive
    ("देखकर", "VRB"),         # conjunctive participle
    ("करेंगे", "VFM"),        # future
    ("राष्ट्रीय", "JJ"),
])
def test_devanagari_suffix_rules_fire(word, expected):
    assert candidate_tags(word)[0] == expected


def test_devanagari_words_do_not_reach_the_english_rules():
    assert candidate_tags("भारत") == ()


def test_english_words_do_not_reach_the_devanagari_rules():
    devanagari_tags = {tags[0] for _, tags in DEVANAGARI_SUFFIX_RULES}
    assert not devanagari_tags & set(candidate_tags("running"))


def test_devanagari_is_unicameral_so_no_proper_noun_rule_applies():
    """The most productive English rule has no Hindi counterpart at all.

    Capitalisation carries the proper-noun signal in English and does not exist
    in Devanagari, so a Hindi proper noun is indistinguishable by shape from
    any other unseen word. Phase 5's accuracy gap starts here.
    """
    assert candidate_tags("दिल्ली", is_first=False) != PROPER_TAGS


def test_short_devanagari_words_do_not_trigger_suffix_rules():
    # Two characters of stem are required, exactly as in English.
    assert candidate_tags("है") == ()
    assert candidate_tags("ने") == ()


def test_danda_is_recognised_as_punctuation():
    for mark in DANDA:
        assert candidate_tags(mark) == PUNCT_TAGS


def test_devanagari_digits_are_recognised_as_numerals():
    assert candidate_tags("१९९८") == NUM_TAGS


def test_the_hindi_tagset_has_a_home_in_the_shape_rules():
    # The BIS spellings, alongside the universal and Penn ones.
    assert "PUNC" in PUNCT_TAGS and "QFNUM" in NUM_TAGS


# ----------------------------------------------------------- the same class

def test_the_same_class_trains_and_decodes_hindi():
    """Phase 5's actual claim: zero code changes, only a different corpus."""
    tagger = hindi_tagger()
    assert set(tagger.tags) == {"NNP", "PREP", "NN", "VFM", "PRP", "JJ",
                                "VAUX", "PUNC"}
    tokens = ["राम", "ने", "किताब", "पढ़ी", "।"]
    assert tagger.viterbi(tokens) == ["NNP", "PREP", "NN", "VFM", "PUNC"]


def test_hindi_context_decides_the_tag_as_it_does_in_english():
    tagged = dict(hindi_tagger().tag(["यह", "अच्छी", "किताब", "है", "।"]))
    assert tagged["किताब"] == "NN"


def test_an_unseen_hindi_word_gets_the_devanagari_heuristic():
    tagger = hindi_tagger()
    assert not tagger.knows("मकानों")
    # Only gaps between tags carry meaning here: _boosted holds the guessed
    # tag at its prior score and pushes every other tag down.
    noun, verb = tagger.tags.index("NN"), tagger.tags.index("VFM")
    prior = tagger.log_emission_unk
    boosted = tagger.emission_column("मकानों")
    assert boosted[noun] - boosted[verb] > prior[noun] - prior[verb]


def test_hindi_model_survives_a_save_load_round_trip(tmp_path):
    tagger = hindi_tagger()
    tokens = ["सीता", "ने", "किताब", "लिखी", "।"]
    restored = HMMTagger.load(tagger.save(tmp_path / "hindi.pkl"))
    assert restored.viterbi(tokens) == tagger.viterbi(tokens)


# --------------------------------------------------------- corpus cleaning

def test_cleaning_drops_tokens_with_a_blank_tag():
    # NLTK's hindi.pos ships a couple of dozen of these. Left in, "" becomes a
    # tag: a row in both matrices that no prediction can ever match.
    cleaned = clean_tagged_sentences([[("घर", "NN"), ("कुछ", "  "), ("है", "VAUX")]])
    assert cleaned == [[("घर", "NN"), ("है", "VAUX")]]


def test_cleaning_drops_a_sentence_left_empty():
    assert clean_tagged_sentences([[("कुछ", "")], [("घर", "NN")]]) == [[("घर", "NN")]]


def test_cleaning_splits_a_glued_danda_off_the_word_in_front_of_it():
    cleaned = clean_tagged_sentences([[("घर।", "NN")]])
    assert [w for w, _ in cleaned[0]] == ["घर", "।"]


def test_a_split_danda_takes_the_tag_this_corpus_gives_punctuation():
    # Read off the corpus, because punctuation is "." in the universal tagset
    # and "PUNC" in the Hindi one.
    cleaned = clean_tagged_sentences(HINDI + [[("घर।", "NN")]])
    assert cleaned[-1] == [("घर", "NN"), ("।", "PUNC")]


def test_a_token_of_nothing_but_dandas_stays_punctuation():
    cleaned = clean_tagged_sentences([[("घर", "NN"), ("।", "PUNC"), ("॥", "PUNC")]])
    assert [t for _, t in cleaned[0]] == ["NN", "PUNC", "PUNC"]


def test_cleaning_normalises_devanagari_to_one_spelling():
    # U+0958 and U+0915 U+093C are one letter written two ways. Unnormalised
    # they are two word types, inflating both vocabulary and OOV rate.
    precomposed = [[("\u0958ानून", "NN")]]
    decomposed = [[("\u0915\u093cानून", "NN")]]
    assert clean_tagged_sentences(precomposed) == clean_tagged_sentences(decomposed)


def test_cleaning_leaves_english_alone():
    assert clean_tagged_sentences(TRAIN) == [list(s) for s in TRAIN]


def test_cleaning_does_not_alias_its_input():
    sentences = [list(s) for s in HINDI]
    clean_tagged_sentences(sentences)[0].append(("injected", "NN"))
    assert all(("injected", "NN") not in s for s in sentences)


# ----------------------------------------------------------- rule reporting

def test_suffix_rule_report_counts_types_not_tokens():
    # These rules only ever meet unseen words, so counting a frequent word once
    # per occurrence would measure a population they never see.
    once = [[("मकानों", "NN")]]
    many = [[("मकानों", "NN")] for _ in range(9)]
    assert suffix_rule_report(once) == suffix_rule_report(many)


def test_suffix_rule_report_scores_the_rule_against_the_gold_tag():
    rows = {r["suffix"]: r for r in suffix_rule_report(
        [[("मकानों", "NN")], [("दिनों", "VFM")]])}
    assert rows["ों"]["types"] == 2
    assert rows["ों"]["purity"] == 0.5


def test_suffix_rule_report_uses_the_taggers_own_first_match_order():
    # "ियों" shadows "ों" here exactly as it does inside candidate_tags.
    assert {r["suffix"] for r in suffix_rule_report([[("बच्चियों", "NN")]])} == {"ियों"}


# ------------------------------------------------------------- corpus shape

def test_corpus_stats_report_size_and_type_token_ratio():
    stats = tagged_corpus_stats(HINDI)
    assert stats["sentences"] == 4
    assert stats["tokens"] == 20
    assert stats["types"] == len({w for s in HINDI for w, _ in s})
    assert stats["ttr"] == pytest.approx(stats["types"] / stats["tokens"])


def test_corpus_stats_on_an_empty_corpus_do_not_divide_by_zero():
    assert tagged_corpus_stats([])["ttr"] == 0.0


# ------------------------------------------------------- real Hindi (slow)

def _hindi(limit: int | None = None):
    try:
        return load_tagged_sentences("hindi", limit=limit)
    except Exception:
        pytest.skip("tagged Hindi corpus not available -- run setup_env.py")


def test_hindi_corpus_loads_as_devanagari_word_tag_pairs():
    sentences = _hindi(limit=20)
    assert any(contains_devanagari(w) for s in sentences for w, _ in s)


def test_the_loaded_hindi_corpus_carries_no_blank_tags():
    assert all(t.strip() for s in _hindi() for _, t in s)


def test_hindi_hmm_beats_the_most_frequent_tag_baseline():
    """Phase 5's acceptance criterion: the identical class works on Hindi."""
    train, test = train_test_split_tagged(_hindi())
    hmm = HMMTagger().fit(train).evaluate(test)
    baseline = MostFrequentTagBaseline().fit(train).evaluate(test)
    assert hmm.accuracy > baseline.accuracy


def test_the_devanagari_rules_earn_their_place_on_unseen_hindi_words():
    train, test = train_test_split_tagged(_hindi())
    rows = {r["strategy"]: r for r in unknown_word_ablation(
        train, test, priors=("hapax",))}
    assert rows["suffix"]["unknown_accuracy"] > rows["uniform"]["unknown_accuracy"]


def test_the_accuracy_gap_between_languages_tracks_the_oov_rate():
    """The second half of the Phase 5 criterion, as a test rather than a claim.

    The gap has to be *explained* by a measured OOV rate rather than asserted,
    so what is pinned down here is the relationship, not a winner: whichever
    language leaves more of its held-out text unseen is the one that scores
    lower, and it is the one with the higher type-token ratio. Were a language
    ever to score lower at the *lower* OOV rate, the report's explanation would
    be wrong and this would say so.

    The direction was English-ahead on NLTK ``indian`` (~10k Hindi tokens) and
    is Hindi-ahead on UD Hindi-HDTB (316k tokens, TTR 0.057) against UD
    English-EWT/GUM (430k tokens, TTR 0.073). Both orderings satisfy this test,
    which is the point: the corpus decides the direction, the OOV rate explains
    it.
    """
    try:
        rows = {r["language"]: r for r in language_comparison()}
    except Exception:
        pytest.skip("tagged corpora not available -- run setup_env.py")
    harder, easier = sorted(rows.values(), key=lambda r: r["oov_rate"], reverse=True)
    assert harder["accuracy"] < easier["accuracy"]
    assert harder["ttr"] > easier["ttr"]


def test_language_comparison_carries_the_fitted_model_along():
    try:
        rows = language_comparison(("hindi",), limit=60)
    except Exception:
        pytest.skip("tagged Hindi corpus not available -- run setup_env.py")
    assert rows[0]["tagger"].is_fitted and rows[0]["test"]
