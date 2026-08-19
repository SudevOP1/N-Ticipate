"""Phase 2 tests — n-gram language model and smoothing."""

from __future__ import annotations

import math

import pytest

from nticipate.ngram import (
    NORMALIZED_METHODS,
    SMOOTHING_METHODS,
    NgramModel,
    perplexity_sweep,
    pruning_report,
    train_model,
)

TRAIN = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "cat", "ate", "the", "fish"],
    ["the", "dog", "sat", "on", "the", "log"],
    ["a", "dog", "ate", "the", "bone"],
    ["the", "cat", "sat", "on", "the", "log"],
]

HELD_OUT = [["the", "cat", "sat", "on", "the", "mat"]]


def make(order=3, smoothing="stupid_backoff", sentences=TRAIN, **kw):
    return NgramModel(order=order, smoothing=smoothing, **kw).fit(sentences)


# ------------------------------------------------------------ construction

def test_rejects_bad_order():
    with pytest.raises(ValueError):
        NgramModel(order=0)


def test_rejects_unknown_smoothing():
    with pytest.raises(ValueError):
        NgramModel(smoothing="magic")


@pytest.mark.parametrize("method", SMOOTHING_METHODS)
def test_every_advertised_method_trains(method):
    assert make(smoothing=method).stats().total_ngrams > 0


def test_repr_mentions_order_and_smoothing():
    text = repr(make(order=2, smoothing="laplace"))
    assert "order=2" in text and "laplace" in text


# --------------------------------------------------------------- counting

def test_unigram_counts():
    model = make(order=1)
    unigrams = model.counts[1][()]
    # "the" appears 9 times across TRAIN.
    assert unigrams["the"] == 9
    assert unigrams["cat"] == 3


def test_bigram_counts():
    model = make(order=2)
    assert model.counts[2][("the",)]["cat"] == 3
    assert model.counts[2][("the",)]["dog"] == 1


def test_trigram_counts():
    model = make(order=3)
    assert model.counts[3][("the", "cat")]["sat"] == 2
    assert model.counts[3][("the", "cat")]["ate"] == 1


def test_bos_padding_gives_order_minus_one_markers():
    model = make(order=3)
    # A trigram model predicting the first real word needs two BOS tokens.
    assert model.counts[3][("<s>", "<s>")]["the"] == 4


def test_eos_is_counted_as_a_predicted_token():
    # The model must learn where sentences end, or generation runs on forever.
    assert make(order=1).counts[1][()]["</s>"] == len(TRAIN)


def test_bos_is_never_predicted():
    assert make(order=1).counts[1][()]["<s>"] == 0


def test_vocab_includes_boundary_symbols():
    model = make()
    assert {"<s>", "</s>"} <= model.vocab


def test_all_orders_are_counted():
    model = make(order=3)
    assert set(model.counts) == {1, 2, 3}


# ---------------------------------------------------------------- scoring

def test_mle_matches_hand_computed_ratio():
    model = make(order=2, smoothing="mle")
    # "the" is followed by cat x3, sat... -> count(the cat)/count(the *)
    counter = model.counts[2][("the",)]
    expected = counter["cat"] / sum(counter.values())
    assert model.prob("cat", ["the"]) == pytest.approx(expected)


def test_mle_gives_zero_to_unseen_ngram():
    # The zero-probability problem, which is the entire reason for smoothing.
    model = make(order=2, smoothing="mle")
    assert model.prob("fish", ["dog"]) == 0.0
    assert model.logprob("fish", ["dog"]) == -math.inf


def test_laplace_never_returns_zero():
    model = make(order=2, smoothing="laplace")
    assert model.prob("fish", ["dog"]) > 0.0


def test_laplace_matches_add_k_formula():
    model = make(order=2, smoothing="laplace", laplace_k=1.0)
    counter = model.counts[2][("the",)]
    total = sum(counter.values())
    expected = (counter["cat"] + 1.0) / (total + 1.0 * len(model.vocab))
    assert model.prob("cat", ["the"]) == pytest.approx(expected)


def test_stupid_backoff_falls_back_and_discounts():
    model = make(order=3, smoothing="stupid_backoff", backoff_alpha=0.4)
    # ("a", "dog") -> "ate" exists as a trigram: no discount applied.
    high = model.prob("ate", ["a", "dog"])
    # This context has no trigram evidence, so the score comes from a lower
    # order and carries the alpha penalty.
    low = model.prob("cat", ["nonexistent", "context"])
    assert high > low > 0.0


def test_kneser_ney_never_returns_zero():
    model = make(order=3, smoothing="kneser_ney")
    assert model.prob("fish", ["nothing", "here"]) > 0.0


def test_kneser_ney_penalises_words_with_one_predecessor():
    # The San Francisco effect: "Francisco" is frequent but follows only "San",
    # so its continuation count is 1 and KN must not predict it after an
    # unrelated context. Raw frequency cannot express that; continuation
    # counts can.
    sentences = [["i", "went", "to", "san", "francisco"]] * 10 + [
        ["i", "saw", "the", "shop"],
        ["we", "like", "the", "park"],
        ["they", "visit", "the", "beach"],
    ]
    kn = NgramModel(order=2, smoothing="kneser_ney").fit(sentences)
    mle = NgramModel(order=2, smoothing="mle").fit(sentences)

    # Raw frequency says "francisco" (10) beats "the" (3) by a wide margin.
    assert mle.counts[1][()]["francisco"] > mle.counts[1][()]["the"]
    # Continuation counts say the opposite: "francisco" follows only "san",
    # while "the" follows saw / like / visit. After an unseen context, where
    # both models must back off, KN prefers the word that actually appears in
    # varied contexts.
    assert kn.prob("francisco", ["unseen"]) < kn.prob("the", ["unseen"])


@pytest.mark.parametrize("method", NORMALIZED_METHODS)
def test_normalized_methods_sum_to_one(method):
    model = make(order=2, smoothing=method)
    assert model.distribution_mass(["the"]) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("method", NORMALIZED_METHODS)
def test_normalized_methods_sum_to_one_on_unseen_context(method):
    model = make(order=3, smoothing=method)
    if method == "mle":
        pytest.skip("MLE assigns no mass at all to an unseen context")
    assert model.distribution_mass(["no", "such"]) == pytest.approx(1.0, abs=1e-9)


def test_stupid_backoff_is_not_a_distribution():
    # Documented honestly rather than quietly reported as a probability: its
    # scores rank correctly but do not sum to one, so its perplexity is not
    # strictly comparable with the other three.
    model = make(order=3, smoothing="stupid_backoff")
    assert not model.is_normalized
    assert model.distribution_mass(["the", "cat"]) != pytest.approx(1.0, abs=1e-6)


def test_is_normalized_flag_matches_the_method():
    for method in SMOOTHING_METHODS:
        model = make(smoothing=method)
        assert model.is_normalized == (method in NORMALIZED_METHODS)


def test_out_of_vocabulary_word_scores_as_unk():
    model = make(order=2, smoothing="laplace")
    assert model.prob("zzzz", ["the"]) == pytest.approx(model.prob("<UNK>", ["the"]))


def test_short_context_is_bos_padded():
    model = make(order=3, smoothing="laplace")
    # One token of context for a trigram model: the missing slot is BOS.
    assert model.prob("cat", ["the"]) == pytest.approx(model.prob("cat", ["<s>", "the"]))


# ------------------------------------------------------------- perplexity

def test_trigram_mle_perplexity_is_infinite_on_unseen_data():
    # The headline demonstration: an unseen trigram makes MLE's perplexity
    # infinite, no matter how good the rest of the model is.
    model = make(order=3, smoothing="mle")
    assert model.perplexity([["a", "cat", "ate", "the", "bone"]]) == math.inf


def test_unigram_mle_perplexity_is_finite_when_vocabulary_is_closed():
    model = make(order=1, smoothing="mle")
    assert math.isfinite(model.perplexity(HELD_OUT))


@pytest.mark.parametrize("method", ["laplace", "stupid_backoff", "kneser_ney"])
def test_smoothed_perplexity_is_finite_on_unseen_data(method):
    model = make(order=3, smoothing=method)
    ppl = model.perplexity([["a", "cat", "ate", "the", "bone"]])
    assert math.isfinite(ppl) and ppl > 0


def test_perplexity_is_lower_on_training_data():
    model = make(order=2, smoothing="laplace")
    assert model.perplexity(TRAIN) < model.perplexity([["a", "fish", "sat"]])


def test_perplexity_of_nothing_is_infinite():
    assert make().perplexity([]) == math.inf


def test_perplexity_does_not_score_bos():
    # Nothing predicts a BOS marker, so scoring it would be meaningless and
    # would make longer-order models look artificially better.
    model = make(order=3, smoothing="laplace")
    assert math.isfinite(model.perplexity([["the", "cat"]]))


# -------------------------------------------------------------- candidates

def test_candidates_are_sorted_by_score():
    scores = [s for _, s in make(order=3).candidates(["the"], k=5)]
    assert scores == sorted(scores, reverse=True)


def test_candidates_respect_k():
    assert len(make(order=3).candidates(["the"], k=2)) == 2


def test_candidates_predict_the_obvious_continuation():
    model = make(order=3, smoothing="stupid_backoff")
    words = [w for w, _ in model.candidates(["the", "cat"], k=3)]
    assert "sat" in words


def test_candidates_never_suggest_bos():
    model = make(order=3)
    assert all(w != "<s>" for w, _ in model.candidates([], k=20))


def test_candidates_never_suggest_unk():
    # <UNK> is a frequent class -- on Brown it outranks every real word after
    # "in the" -- so leaving it in makes the app's top suggestion a literal
    # "<UNK>".
    sentences = [["the", "rare", w] for w in "abcdefghij"] + [
        ["the", "common", "word"]
    ] * 3
    model = NgramModel(order=2, smoothing="stupid_backoff").fit(
        [[t if t in {"the", "common", "word", "rare"} else "<UNK>" for t in s]
         for s in sentences]
    )
    assert all(w != "<UNK>" for w, _ in model.candidates(["the"], k=20))


def test_candidates_honour_exclude():
    model = make(order=3)
    words = [w for w, _ in model.candidates(["the"], k=10, exclude={"cat"})]
    assert "cat" not in words


def test_candidates_on_unseen_context_backs_off():
    # An unseen context must still produce suggestions, or the app goes blank
    # exactly when the user is typing something novel.
    assert make(order=3).candidates(["zzz", "qqq"], k=3)


# --------------------------------------------------------------- generate

def test_generate_produces_tokens():
    assert make(order=2).generate(max_length=10, seed=1)


def test_generate_is_deterministic_for_a_seed():
    model = make(order=3)
    assert model.generate(seed=7) == model.generate(seed=7)


def test_generate_respects_max_length():
    assert len(make(order=1).generate(max_length=5, seed=3)) <= 5


def test_generate_never_emits_boundary_markers():
    tokens = make(order=2).generate(max_length=30, seed=11)
    assert "<s>" not in tokens and "</s>" not in tokens


def test_generate_from_a_given_context():
    model = make(order=3)
    assert isinstance(model.generate(context=["the", "cat"], seed=2), list)


# ---------------------------------------------------------------- pruning

def test_prune_shrinks_the_model():
    model = make(order=3)
    before = model.stats().total_ngrams
    model.prune(min_count=2, max_continuations=0)
    assert model.stats().total_ngrams < before


def test_prune_keeps_unigrams_intact():
    # Unigrams are the vocabulary and the final backoff level; pruning them
    # would put holes in the distribution rather than shrink it.
    model = make(order=3)
    before = dict(model.counts[1][()])
    model.prune(min_count=5, max_continuations=1)
    assert dict(model.counts[1][()]) == before


def test_prune_caps_continuations_per_context():
    model = make(order=3)
    model.prune(min_count=1, max_continuations=1)
    assert all(len(c) <= 1 for c in model.counts[2].values())


def test_prune_sets_the_flag():
    model = make()
    assert not model.pruned
    model.prune()
    assert model.pruned


def test_pruned_model_still_scores_every_word():
    model = make(order=3, smoothing="stupid_backoff")
    model.prune(min_count=3, max_continuations=1)
    assert model.prob("cat", ["the"]) > 0


def test_prune_rebuilds_continuation_counts():
    model = make(order=3, smoothing="kneser_ney")
    model.prune(min_count=2, max_continuations=0)
    assert math.isfinite(model.perplexity(HELD_OUT))


# --------------------------------------------------------- serialisation

def test_save_load_round_trip(tmp_path):
    model = make(order=3, smoothing="kneser_ney")
    restored = NgramModel.load(model.save(tmp_path / "m.pkl"))
    assert restored.order == model.order
    assert restored.smoothing == model.smoothing
    assert restored.vocab == model.vocab
    assert restored.stats().total_ngrams == model.stats().total_ngrams


def test_save_load_preserves_scores(tmp_path):
    model = make(order=3, smoothing="kneser_ney")
    restored = NgramModel.load(model.save(tmp_path / "m.pkl"))
    assert restored.prob("sat", ["the", "cat"]) == pytest.approx(
        model.prob("sat", ["the", "cat"])
    )


def test_save_load_preserves_perplexity(tmp_path):
    model = make(order=3, smoothing="laplace")
    restored = NgramModel.load(model.save(tmp_path / "m.pkl"))
    assert restored.perplexity(HELD_OUT) == pytest.approx(model.perplexity(HELD_OUT))


def test_loaded_model_can_still_be_pruned(tmp_path):
    model = NgramModel.load(make(order=3).save(tmp_path / "m.pkl"))
    model.prune(min_count=2, max_continuations=0)
    assert model.pruned


# ------------------------------------------------------------ experiments

def test_train_model_helper():
    assert train_model(TRAIN, 2, "laplace").order == 2


def test_perplexity_sweep_covers_every_pair():
    rows = perplexity_sweep(TRAIN, HELD_OUT, orders=(1, 2), methods=("mle", "laplace"))
    assert len(rows) == 4
    assert {(r["order"], r["smoothing"]) for r in rows} == {
        (1, "mle"), (1, "laplace"), (2, "mle"), (2, "laplace")
    }


def test_perplexity_sweep_flags_unnormalized_rows():
    rows = perplexity_sweep(TRAIN, HELD_OUT, orders=(2,), methods=("stupid_backoff",))
    assert rows[0]["normalized"] is False


def test_pruning_report_shrinks_and_reports_size():
    rows = pruning_report(TRAIN, HELD_OUT, order=3, settings=((1, 0), (2, 0)))
    assert rows[0]["ngrams"] > rows[1]["ngrams"]
    assert rows[0]["size_mb"] > 0


# ------------------------------------------------------------------ stats

def test_stats_reports_per_order_counts():
    stats = make(order=3).stats()
    assert set(stats.ngram_counts) == {1, 2, 3}
    assert stats.total_ngrams == sum(stats.ngram_counts.values())
    assert stats.vocab_size == len(make(order=3).vocab)


# ------------------------------------------------------------ totals cache

def test_totals_cache_matches_recomputed_sums():
    model = make(order=3)
    for k, table in model.counts.items():
        for ctx, counter in table.items():
            assert model.totals[k][ctx] == sum(counter.values())


def test_totals_cache_is_rebuilt_after_pruning():
    model = make(order=3)
    model.prune(min_count=2, max_continuations=0)
    for k, table in model.counts.items():
        for ctx, counter in table.items():
            assert model.totals[k][ctx] == sum(counter.values())


def test_totals_cache_is_rebuilt_after_loading(tmp_path):
    model = NgramModel.load(make(order=3).save(tmp_path / "m.pkl"))
    assert model.totals[1][()] == sum(model.counts[1][()].values())
