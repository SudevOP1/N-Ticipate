"""Phase 3 tests — the prediction engine."""

from __future__ import annotations

import pytest

from nticipate.ngram import NgramModel
from nticipate.predictor import (
    Mode,
    Predictor,
    completion_hit_at_k,
    hit_at_k,
    keystroke_savings,
    latency_stats,
)
from nticipate.trie import Trie
from nticipate.userprofile import UserProfile

TRAIN = [
    ["i", "would", "like", "to", "know", "more"],
    ["i", "would", "like", "to", "know", "why"],
    ["i", "would", "like", "to", "thank", "you"],
    ["the", "recent", "report", "was", "good"],
    ["the", "record", "was", "broken"],
    ["please", "recommend", "a", "good", "book"],
    ["the", "report", "was", "good"],
] * 3


def make_predictor(profile=None, **kw) -> Predictor:
    model = NgramModel(order=3, smoothing="stupid_backoff").fit(TRAIN)
    kw.setdefault("truecase", {"i": "I"})
    return Predictor(model, profile=profile, **kw)


# --------------------------------------------------------------- construction

def test_trie_is_built_from_the_model():
    predictor = make_predictor()
    assert "recommend" in predictor.trie
    assert len(predictor.trie) > 0


def test_trie_excludes_boundary_markers_and_unk():
    # None of these are words a user can type a prefix of, and <UNK> is a
    # frequent class that would dominate every completion list.
    predictor = make_predictor()
    for special in ("<s>", "</s>", "<UNK>"):
        assert special not in predictor.trie


def test_explicit_trie_is_used():
    model = NgramModel(order=2, smoothing="stupid_backoff").fit(TRAIN)
    trie = Trie(["custom"])
    assert Predictor(model, trie=trie).trie is trie


def test_personalization_can_be_disabled():
    assert make_predictor(personalization=False).profile is None


def test_context_size_is_order_minus_one():
    assert make_predictor().context_size == 2


def test_repr_mentions_order_and_lambda():
    text = repr(make_predictor())
    assert "order=3" in text and "lambda=" in text


# -------------------------------------------------------------- buffer split

@pytest.mark.parametrize("text,context,prefix", [
    ("i would like to ", ["like", "to"], ""),
    ("i would like to kn", ["like", "to"], "kn"),
    ("hello", [], "hello"),
    ("", [], ""),
    ("   ", [], ""),
])
def test_split_buffer(text, context, prefix):
    predictor = make_predictor()
    got_context, got_prefix = predictor.split_buffer(text)
    assert got_context == context
    assert got_prefix == prefix


def test_split_buffer_keeps_only_the_models_context_window():
    predictor = make_predictor()
    context, _ = predictor.split_buffer("one two three four five ")
    assert len(context) == predictor.context_size


# ------------------------------------------------------------------- modes

def test_empty_prefix_is_next_word_mode():
    suggestions = make_predictor().predict(["like", "to"], "")
    assert suggestions and all(s.mode is Mode.NEXT_WORD for s in suggestions)


def test_non_empty_prefix_is_completion_mode():
    suggestions = make_predictor().predict(["please"], "rec")
    assert suggestions and all(s.mode is Mode.COMPLETION for s in suggestions)


def test_next_word_prediction_is_sensible():
    words = [s.word.lower() for s in make_predictor().predict(["like", "to"], "", k=3)]
    assert "know" in words


def test_completion_narrows_to_the_prefix():
    words = [s.word.lower() for s in make_predictor().predict([], "rec", k=5)]
    assert words and all(w.startswith("rec") for w in words)


def test_short_prefix_below_minimum_returns_nothing():
    predictor = make_predictor(min_prefix_len=3)
    assert predictor.predict([], "r") == []


def test_suggestions_are_ranked_by_score():
    scores = [s.score for s in make_predictor().predict(["like", "to"], "", k=5)]
    assert scores == sorted(scores, reverse=True)


def test_k_limits_the_result():
    assert len(make_predictor().predict(["like", "to"], "", k=2)) == 2


def test_default_k_is_max_suggestions():
    predictor = make_predictor(max_suggestions=2)
    assert len(predictor.predict(["like", "to"], "")) == 2


def test_suggestions_never_include_specials():
    predictor = make_predictor()
    words = [s.word for s in predictor.predict([], "", k=20)]
    assert not ({"<s>", "</s>", "<UNK>"} & set(words))


def test_unseen_context_still_suggests():
    # The app must not go blank exactly when the user types something novel.
    assert make_predictor().predict(["zzz", "qqq"], "", k=3)


def test_suggestions_are_truecased():
    predictor = make_predictor(truecase={"know": "Know"})
    words = [s.word for s in predictor.predict(["like", "to"], "", k=3)]
    assert "Know" in words


def test_suggest_from_a_raw_buffer():
    words = [s.word.lower() for s in make_predictor().suggest("i would like to ")]
    assert "know" in words


def test_suggest_handles_completion_from_a_raw_buffer():
    words = [s.word.lower() for s in make_predictor().suggest("please rec")]
    assert any(w.startswith("rec") for w in words)


# --------------------------------------------------------- personalisation

def test_fresh_profile_does_not_change_ranking():
    # lambda is 0 with no evidence, so a fresh install is exactly the base model.
    base = make_predictor(personalization=False)
    fresh = make_predictor(profile=UserProfile())
    assert [s.word for s in base.predict(["like", "to"], "")] == \
           [s.word for s in fresh.predict(["like", "to"], "")]


def test_user_words_become_completable():
    # The headline personalisation claim: words the base corpus never saw.
    profile = UserProfile(lambda_max=0.4, lambda_growth_tokens=10)
    predictor = make_predictor(profile=profile)
    assert not [s for s in predictor.predict([], "ntic", k=5)]

    predictor.learn("nticipate is my project. nticipate uses ngrams.")
    words = [s.word.lower() for s in predictor.predict([], "ntic", k=5)]
    assert "nticipate" in words


def test_user_model_lifts_a_learned_continuation():
    profile = UserProfile(lambda_max=0.9, lambda_growth_tokens=5)
    predictor = make_predictor(profile=profile)
    before = [s.word.lower() for s in predictor.predict(["like", "to"], "", k=3)]
    assert "deploy" not in before

    for _ in range(10):
        profile.observe(["i", "would", "like", "to", "deploy"])
    after = [s.word.lower() for s in predictor.predict(["like", "to"], "", k=3)]
    assert "deploy" in after


def test_suggestions_report_their_source():
    profile = UserProfile(lambda_max=0.5, lambda_growth_tokens=5)
    predictor = make_predictor(profile=profile)
    for _ in range(5):
        profile.observe(["ship", "nticipate", "today"])
    sources = {s.source for s in predictor.predict(["ship"], "", k=5)}
    assert "user" in sources


def test_learn_returns_token_count():
    predictor = make_predictor(profile=UserProfile())
    assert predictor.learn("hello world") == 2


def test_learn_without_a_profile_is_a_noop():
    assert make_predictor(personalization=False).learn("anything") == 0


# ------------------------------------------------------------- evaluation

HELD_OUT = [
    ["i", "would", "like", "to", "know", "more"],
    ["the", "report", "was", "good"],
]


def test_hit_at_k_is_monotonic_in_k():
    result = hit_at_k(make_predictor(), HELD_OUT, ks=(1, 3, 5))
    assert result["hit@1"] <= result["hit@3"] <= result["hit@5"]
    assert result["positions"] > 0


def test_hit_at_k_finds_learned_continuations():
    assert hit_at_k(make_predictor(), HELD_OUT, ks=(5,))["hit@5"] > 0


def test_hit_at_k_ignores_unk_targets():
    # The predictor never suggests <UNK>, so scoring those positions would
    # report misses for words the engine is designed not to offer.
    with_unk = hit_at_k(make_predictor(), [["the", "<UNK>", "was", "good"]], ks=(1,))
    without = hit_at_k(make_predictor(), [["the", "was", "good"]], ks=(1,))
    assert with_unk["positions"] == without["positions"]


def test_completion_hit_at_k_beats_next_word():
    # Two characters of evidence should help; if it does not, something is
    # wrong with the trie path.
    predictor = make_predictor()
    next_word = hit_at_k(predictor, HELD_OUT, ks=(3,))["hit@3"]
    completion = completion_hit_at_k(predictor, HELD_OUT, ks=(3,), prefix_len=2)["hit@3"]
    assert completion >= next_word


def test_completion_skips_words_shorter_than_the_prefix():
    result = completion_hit_at_k(make_predictor(), [["a", "an", "the"]], prefix_len=3)
    assert result["positions"] == 0


def test_keystroke_savings_reports_a_fraction():
    result = keystroke_savings(make_predictor(), HELD_OUT, k=3)
    assert 0.0 <= result["savings"] < 1.0
    assert result["words"] > 0
    assert result["keystrokes_typed"] <= result["keystrokes_baseline"]


def test_keystroke_savings_never_accepts_a_losing_suggestion():
    # Accepting a 3-letter word after typing 2 letters saves nothing, and a
    # real user would not do it.
    result = keystroke_savings(make_predictor(), [["was"]], k=5)
    assert result["keystrokes_typed"] <= result["keystrokes_baseline"]


def test_keystroke_savings_on_nothing():
    assert keystroke_savings(make_predictor(), [])["words"] == 0


def test_latency_stats_reports_percentiles():
    predictor = make_predictor()
    stats = latency_stats(predictor, [(["like", "to"], ""), ([], "rec")], repeats=5)
    assert stats["calls"] == 10
    assert stats["p50_ms"] > 0 and stats["p95_ms"] >= stats["p50_ms"]


def test_latency_stats_on_nothing():
    assert latency_stats(make_predictor(), [])["calls"] == 0


# ------------------------------------------------------- punctuation filter

def test_punctuation_is_suggested_by_default():
    # Default on, so the Phase 3 metrics measure the model itself rather than
    # a display policy layered over it.
    model = NgramModel(order=2, smoothing="stupid_backoff").fit(
        [["good", ".", "good", ",", "good", "!"]] * 5
    )
    predictor = Predictor(model, personalization=False)
    words = [s.word for s in predictor.predict(["good"], "", k=5)]
    assert any(w in {".", ",", "!"} for w in words)


def test_punctuation_can_be_filtered_out():
    # As a *suggestion* "." is useless: the user types it faster than they can
    # read it. Phase 7 turns this on for display.
    model = NgramModel(order=2, smoothing="stupid_backoff").fit(
        [["good", ".", "good", ",", "good", "!"]] * 5
    )
    predictor = Predictor(model, personalization=False, suggest_punctuation=False)
    words = [s.word for s in predictor.predict(["good"], "", k=5)]
    assert not any(w in {".", ",", "!"} for w in words)


def test_punctuation_filter_keeps_real_words():
    predictor = make_predictor(suggest_punctuation=False)
    words = [s.word.lower() for s in predictor.predict(["like", "to"], "", k=3)]
    assert "know" in words
