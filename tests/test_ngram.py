"""
Phase 2 tests. Uses tiny hand-constructed corpora with known counts so
expected probabilities can be checked exactly (MLE, Laplace) or by
inequality (stupid_backoff, Kneser-Ney, where the closed form is more
involved).

Run with: pytest tests/test_ngram.py -v
"""

import math

from nticipate.preprocess import START_TOKEN, END_TOKEN
from nticipate.ngram import NgramModel, build_ngram_hierarchy


# A tiny, hand-countable corpus:
#   "the cat sat" x2, "the dog sat" x1
# Padded (n=2): <s> the cat sat </s>  (x2)
#               <s> the dog sat </s>  (x1)
CORPUS = [
    ["the", "cat", "sat"],
    ["the", "cat", "sat"],
    ["the", "dog", "sat"],
]
VOCAB = {"the", "cat", "dog", "sat", "<UNK>", START_TOKEN, END_TOKEN}


# ---------------------------------------------------------------------------
# fit() / counting
# ---------------------------------------------------------------------------
def test_fit_bigram_counts_are_correct():
    model = NgramModel(n=2, smoothing="mle").fit(CORPUS, vocab=VOCAB)
    # "the" appears as context for cat (x2) and dog (x1) -> total 3
    assert model._context_total(("the",)) == 3
    assert model._context_word_count(("the",), "cat") == 2
    assert model._context_word_count(("the",), "dog") == 1


def test_fit_pads_correctly_for_trigram():
    model = NgramModel(n=3, smoothing="mle").fit(CORPUS, vocab=VOCAB)
    # padded: <s> <s> the cat sat </s> -- context (<s>,<s>) should precede "the"
    assert model._context_word_count((START_TOKEN, START_TOKEN), "the") == 3


# ---------------------------------------------------------------------------
# MLE
# ---------------------------------------------------------------------------
def test_mle_matches_hand_computed_probability():
    model = NgramModel(n=2, smoothing="mle").fit(CORPUS, vocab=VOCAB)
    # P(cat | the) = 2/3
    assert math.isclose(model.prob("cat", ("the",)), 2 / 3)
    # P(dog | the) = 1/3
    assert math.isclose(model.prob("dog", ("the",)), 1 / 3)


def test_mle_zero_for_unseen_bigram():
    model = NgramModel(n=2, smoothing="mle").fit(CORPUS, vocab=VOCAB)
    assert model.prob("sat", ("dog", "the")) == 0.0  # not even a valid context shape, but should just be 0
    assert model.prob("mat", ("the",)) == 0.0  # "mat" never follows "the" in this corpus


# ---------------------------------------------------------------------------
# Laplace
# ---------------------------------------------------------------------------
def test_laplace_matches_hand_computed_probability():
    from nticipate.ngram import CFG

    k = CFG["ngram"]["add_k"]
    v = len(VOCAB)
    model = NgramModel(n=2, smoothing="laplace").fit(CORPUS, vocab=VOCAB)
    expected = (2 + k) / (3 + k * v)
    assert math.isclose(model.prob("cat", ("the",)), expected)


def test_laplace_never_returns_exact_zero():
    model = NgramModel(n=2, smoothing="laplace").fit(CORPUS, vocab=VOCAB)
    assert model.prob("mat", ("the",)) > 0.0


# ---------------------------------------------------------------------------
# stupid backoff
# ---------------------------------------------------------------------------
def test_stupid_backoff_uses_direct_count_when_available():
    models = build_ngram_hierarchy(CORPUS, orders=[1, 2], smoothing="stupid_backoff", vocab=VOCAB)
    bigram = models[2]
    # seen bigram -> should equal plain MLE-style count/total, no backoff needed
    assert math.isclose(bigram.prob("cat", ("the",)), 2 / 3)


def test_stupid_backoff_falls_back_for_unseen_context():
    models = build_ngram_hierarchy(CORPUS, orders=[1, 2], smoothing="stupid_backoff", vocab=VOCAB)
    bigram = models[2]
    # "sat" never seen after "cat" directly as the queried context in this
    # tiny corpus's bigram counts for word "the" -- check an actually
    # unseen (context, word) pair backs off to a non-zero value
    p = bigram.prob("dog", ("sat",))  # "dog" never follows "sat"
    assert p > 0.0  # backed off, not hard zero


def test_stupid_backoff_never_exceeds_one():
    models = build_ngram_hierarchy(CORPUS, orders=[1, 2, 3], smoothing="stupid_backoff", vocab=VOCAB)
    for model in models.values():
        for word in VOCAB:
            assert 0.0 <= model.prob(word, ("the", "cat")) <= 1.0001


# ---------------------------------------------------------------------------
# Kneser-Ney
# ---------------------------------------------------------------------------
def test_kneser_ney_probabilities_are_valid_range():
    models = build_ngram_hierarchy(CORPUS, orders=[1, 2, 3], smoothing="kneser_ney", vocab=VOCAB)
    trigram = models[3]
    for word in VOCAB:
        p = trigram.prob(word, ("the", "cat"))
        assert 0.0 <= p <= 1.0001


def test_kneser_ney_unigram_uses_continuation_counts():
    models = build_ngram_hierarchy(CORPUS, orders=[1, 2], smoothing="kneser_ney", vocab=VOCAB)
    unigram = models[1]
    # "sat" follows two distinct words ("cat", "dog") -> continuation count 2
    # "cat" and "dog" each follow only "the" -> continuation count 1 each
    assert unigram._kn_unigram_continuation["sat"] == 2
    assert unigram._kn_unigram_continuation["cat"] == 1


# ---------------------------------------------------------------------------
# perplexity
# ---------------------------------------------------------------------------
def test_perplexity_is_lower_for_seen_data_than_random_data():
    model = NgramModel(n=2, smoothing="laplace").fit(CORPUS, vocab=VOCAB)
    pp_train = model.perplexity(CORPUS)
    pp_random = model.perplexity([["dog", "dog", "dog", "cat", "cat"]])
    assert pp_train < pp_random


def test_perplexity_mle_is_finite_due_to_epsilon_floor():
    model = NgramModel(n=2, smoothing="mle").fit(CORPUS, vocab=VOCAB)
    # this sentence contains a bigram never seen in training ("dog", "cat")
    pp = model.perplexity([["dog", "cat"]])
    assert pp < float("inf")
    assert pp > 1.0  # should be a large but finite number, not near 1


def test_perplexity_empty_input_returns_inf():
    model = NgramModel(n=2, smoothing="mle").fit(CORPUS, vocab=VOCAB)
    assert model.perplexity([]) == float("inf")


# ---------------------------------------------------------------------------
# top_k
# ---------------------------------------------------------------------------
def test_top_k_returns_sorted_descending():
    model = NgramModel(n=2, smoothing="mle").fit(CORPUS, vocab=VOCAB)
    results = model.top_k(("the",), k=5)
    probs = [p for _, p in results]
    assert probs == sorted(probs, reverse=True)


def test_top_k_respects_k_limit():
    model = NgramModel(n=2, smoothing="laplace").fit(CORPUS, vocab=VOCAB)
    results = model.top_k(("the",), k=2)
    assert len(results) <= 2


def test_top_k_excludes_start_token():
    model = NgramModel(n=2, smoothing="laplace").fit(CORPUS, vocab=VOCAB)
    results = model.top_k(("the",), k=20)
    assert all(w != START_TOKEN for w, _ in results)


# ---------------------------------------------------------------------------
# pruning
# ---------------------------------------------------------------------------
def test_prune_drops_rare_continuations():
    model = NgramModel(n=2, smoothing="mle").fit(CORPUS, vocab=VOCAB)
    model.prune(min_count=2, top_k_per_context=10)
    # "dog" only occurred once after "the" -- should be pruned
    assert model._context_word_count(("the",), "dog") == 0
    # "cat" occurred twice -- should survive
    assert model._context_word_count(("the",), "cat") == 2


def test_prune_caps_continuations_per_context():
    model = NgramModel(n=2, smoothing="mle").fit(CORPUS, vocab=VOCAB)
    model.prune(min_count=1, top_k_per_context=1)
    assert len(model.counts.get(("the",), {})) <= 1


def test_model_size_estimate_returns_expected_keys():
    model = NgramModel(n=2, smoothing="mle").fit(CORPUS, vocab=VOCAB)
    stats = model.model_size_estimate()
    assert set(stats.keys()) == {"num_contexts", "num_entries", "size_bytes"}
    assert stats["size_bytes"] > 0


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------
def test_generate_returns_list_of_strings():
    import random

    model = NgramModel(n=2, smoothing="laplace").fit(CORPUS, vocab=VOCAB)
    tokens = model.generate(max_len=10, rng=random.Random(0))
    assert isinstance(tokens, list)
    assert all(isinstance(t, str) for t in tokens)
    assert END_TOKEN not in tokens  # generation stops at, doesn't include, </s>


def test_generate_is_reproducible_with_same_seed():
    import random

    model = NgramModel(n=2, smoothing="laplace").fit(CORPUS, vocab=VOCAB)
    out1 = model.generate(max_len=10, rng=random.Random(42))
    out2 = model.generate(max_len=10, rng=random.Random(42))
    assert out1 == out2


# ---------------------------------------------------------------------------
# build_ngram_hierarchy wiring
# ---------------------------------------------------------------------------
def test_hierarchy_wires_lower_order_model_chain():
    models = build_ngram_hierarchy(CORPUS, orders=[1, 2, 3], smoothing="stupid_backoff", vocab=VOCAB)
    assert models[1].lower_order_model is None
    assert models[2].lower_order_model is models[1]
    assert models[3].lower_order_model is models[2]


def test_hierarchy_installs_kn_continuation_base():
    models = build_ngram_hierarchy(CORPUS, orders=[1, 2], smoothing="kneser_ney", vocab=VOCAB)
    assert models[1]._kn_unigram_continuation is not None
