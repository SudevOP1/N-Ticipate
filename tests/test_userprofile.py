"""
Phase 3 tests: UserProfile (personalization layer).

Run with: pytest tests/test_userprofile.py -v
"""

import os
import tempfile

from nticipate.userprofile import UserProfile
from nticipate.config import load_config

CFG = load_config()


# ---------------------------------------------------------------------------
# observing sentences / vocab
# ---------------------------------------------------------------------------
def test_observe_sentence_updates_token_count():
    profile = UserProfile()
    profile.observe_sentence(["hello", "world"])
    assert profile.token_count == 2
    profile.observe_sentence(["foo", "bar", "baz"])
    assert profile.token_count == 5


def test_observe_sentence_ignores_empty_input():
    profile = UserProfile()
    profile.observe_sentence([])
    assert profile.token_count == 0
    assert profile.sentences == []


def test_vocab_reflects_observed_tokens_with_original_casing():
    profile = UserProfile()
    profile.observe_sentence(["The", "India", "trip"])
    assert profile.vocab == {"The", "India", "trip"}


# ---------------------------------------------------------------------------
# lazy fitting / querying
# ---------------------------------------------------------------------------
def test_prob_and_top_k_before_any_data_are_safe():
    profile = UserProfile()
    assert profile.prob("hello", ()) == 0.0
    assert profile.top_k(()) == []


def test_prob_reflects_observed_bigram():
    profile = UserProfile()
    for _ in range(5):
        profile.observe_sentence(["i", "love", "pizza"])
    # after fitting, "pizza" should be a likely continuation of "love"
    p = profile.prob("pizza", ("love",), n=2)
    assert p > 0.0


def test_top_k_returns_words_from_observed_data():
    profile = UserProfile()
    for _ in range(5):
        profile.observe_sentence(["i", "love", "pizza"])
    results = profile.top_k(("love",), k=3, n=2)
    words = [w for w, _ in results]
    assert "pizza" in words


# ---------------------------------------------------------------------------
# current_lambda schedule
# ---------------------------------------------------------------------------
def test_lambda_starts_at_lambda_start_below_min_tokens():
    profile = UserProfile()
    profile.token_count = 10  # well below min_user_tokens (200)
    expected = CFG["predictor"]["personalization"]["lambda_start"]
    assert profile.current_lambda() == expected


def test_lambda_at_exactly_min_tokens_is_still_lambda_start():
    profile = UserProfile()
    profile.token_count = CFG["predictor"]["personalization"]["min_user_tokens"]
    expected = CFG["predictor"]["personalization"]["lambda_start"]
    assert profile.current_lambda() == expected


def test_lambda_saturates_at_lambda_max_for_large_token_counts():
    profile = UserProfile()
    profile.token_count = CFG["predictor"]["personalization"]["min_user_tokens"] * 100
    expected = CFG["predictor"]["personalization"]["lambda_max"]
    assert profile.current_lambda() == expected


def test_lambda_increases_monotonically_with_token_count():
    profile = UserProfile()
    min_tokens = CFG["predictor"]["personalization"]["min_user_tokens"]
    checkpoints = [min_tokens, min_tokens * 2, min_tokens * 5, min_tokens * 10]
    lambdas = []
    for tc in checkpoints:
        profile.token_count = tc
        lambdas.append(profile.current_lambda())
    assert lambdas == sorted(lambdas)


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------
def test_save_and_load_round_trip():
    profile = UserProfile()
    for _ in range(3):
        profile.observe_sentence(["hello", "world"])

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "profile.pkl")
        profile.save(path)
        loaded = UserProfile.load(path)

    assert loaded.token_count == profile.token_count
    assert loaded.sentences == profile.sentences
    # loaded profile should still answer queries correctly after refitting
    assert loaded.prob("world", ("hello",), n=2) > 0.0
