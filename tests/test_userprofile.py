"""Phase 3 tests — per-user personalisation layer."""

from __future__ import annotations

import pytest

from nticipate.userprofile import UserProfile


def make(**kw) -> UserProfile:
    kw.setdefault("order", 3)
    return UserProfile(**kw)


# ----------------------------------------------------------------- observing

def test_observe_counts_tokens():
    profile = make()
    assert profile.observe(["hello", "world"]) == 2
    assert profile.token_count == 2


def test_observe_text_tokenises():
    profile = make()
    profile.observe_text("Hello world. Goodbye world.")
    assert profile.token_count > 0
    assert "world" in profile.trie


def test_observe_is_case_insensitive():
    profile = make()
    profile.observe(["Sudev", "wrote", "this"])
    assert "sudev" in profile.trie
    assert profile.prob("sudev", ["<s>", "<s>"]) > 0


def test_observe_empty_sentence_is_a_noop():
    profile = make()
    assert profile.observe([]) == 0
    assert profile.token_count == 0


def test_learns_ngram_structure():
    profile = make()
    for _ in range(5):
        profile.observe(["deploy", "to", "staging"])
    assert profile.prob("staging", ["deploy", "to"]) > 0


def test_user_vocabulary_is_never_unked():
    # The base model discards anything below min_token_freq -- which is
    # exactly the user's colleagues' names and project jargon. Capturing
    # those is the whole point of this layer.
    profile = make()
    profile.observe(["ping", "nticipate", "maintainer"])
    assert "nticipate" in profile.trie
    assert profile.prob("nticipate", ["ping"]) > 0


# ------------------------------------------------------------------ scoring

def test_prob_is_zero_for_unseen_word():
    # A hard zero, so the predictor's interpolation falls back entirely to the
    # base model -- correct for a profile holding a few thousand tokens.
    assert make().prob("unseen") == 0.0


def test_prob_prefers_the_longest_matching_context():
    profile = make()
    for _ in range(3):
        profile.observe(["run", "the", "tests"])
    profile.observe(["skip", "the", "docs"])
    assert profile.prob("tests", ["run", "the"]) > profile.prob("docs", ["run", "the"])


def test_prob_backs_off_to_shorter_context():
    profile = make()
    for _ in range(3):
        profile.observe(["deploy", "to", "production"])
    # No trigram evidence for this context, but the unigram is known.
    assert profile.prob("production", ["completely", "different"]) > 0


def test_candidates_exclude_boundary_markers():
    profile = make()
    profile.observe(["alpha", "beta"])
    words = [w for w, _ in profile.candidates([], k=10)]
    assert "<s>" not in words and "</s>" not in words


def test_candidates_are_sorted():
    profile = make()
    for _ in range(3):
        profile.observe(["ship", "it"])
    profile.observe(["ship", "later"])
    scores = [s for _, s in profile.candidates(["ship"], k=5)]
    assert scores == sorted(scores, reverse=True)


def test_complete_finds_user_words():
    profile = make()
    profile.observe(["nticipate", "notebook"])
    assert set(w for w, _ in profile.complete("not", k=5)) == {"notebook"}


# ------------------------------------------------------------------- weight

def test_weight_starts_at_zero():
    # A fresh install must behave exactly like the base model.
    assert make().weight == 0.0


def test_weight_grows_with_evidence():
    profile = make(lambda_max=0.4, lambda_growth_tokens=100)
    profile.observe(["a"] * 10)
    early = profile.weight
    profile.observe(["b"] * 40)
    assert profile.weight > early > 0.0


def test_weight_is_capped_at_lambda_max():
    # The user's text is small and topically narrow; letting it dominate would
    # make the app worse at ordinary English.
    profile = make(lambda_max=0.4, lambda_growth_tokens=10)
    profile.observe(["x"] * 500)
    assert profile.weight == pytest.approx(0.4)


def test_weight_is_lambda_max_when_growth_disabled():
    profile = make(lambda_max=0.3, lambda_growth_tokens=0)
    assert profile.weight == pytest.approx(0.3)


# ------------------------------------------------------------------ eviction

def test_ring_buffer_evicts_oldest_sentences():
    profile = make(max_tokens=10)
    for i in range(20):
        profile.observe([f"word{i}", "filler"])
    assert profile.token_count <= 10
    assert len(profile.sentences) < 20


def test_eviction_subtracts_counts():
    profile = make(max_tokens=6)
    profile.observe(["old", "old", "old"])
    for _ in range(4):
        profile.observe(["new", "new", "new"])
    # "old" has aged out of the n-gram counts entirely.
    assert profile.prob("old", []) == 0.0
    assert profile.prob("new", []) > 0.0


def test_eviction_keeps_trie_entries():
    # A name typed once should stay completable after its n-gram counts age
    # out. A few bytes, and the app does not forget a colleague mid-chat.
    profile = make(max_tokens=4)
    profile.observe(["priyanka", "reviewed"])
    for _ in range(5):
        profile.observe(["filler", "filler"])
    assert "priyanka" in profile.trie


def test_totals_stay_consistent_after_eviction():
    profile = make(max_tokens=8)
    for i in range(10):
        profile.observe([f"w{i}", "tail"])
    for k, table in profile.counts.items():
        for ctx, counter in table.items():
            assert profile.totals[k][ctx] == sum(counter.values())
            assert all(c > 0 for c in counter.values())


def test_no_eviction_when_under_cap():
    profile = make(max_tokens=1000)
    profile.observe(["a", "b", "c"])
    assert len(profile.sentences) == 1


# ----------------------------------------------------------------- lifecycle

def test_reset_forgets_everything():
    profile = make()
    profile.observe(["secret", "project", "name"])
    profile.reset()
    assert profile.token_count == 0
    assert len(profile.trie) == 0
    assert profile.prob("secret") == 0.0


def test_save_load_round_trip(tmp_path):
    profile = make()
    profile.observe_text("Deploy to staging. Deploy to production.")
    restored = UserProfile.load(profile.save(tmp_path / "p.json"))
    assert restored.token_count == profile.token_count
    assert restored.prob("staging", ["deploy", "to"]) == pytest.approx(
        profile.prob("staging", ["deploy", "to"])
    )


def test_save_stores_only_sentences(tmp_path):
    # Counts and trie are recomputed on load, so a change to the counting
    # logic cannot leave stale derived state on disk.
    import json

    profile = make()
    profile.observe(["a", "b"])
    with open(profile.save(tmp_path / "p.json"), encoding="utf-8") as fh:
        data = json.load(fh)
    assert "sentences" in data
    assert "counts" not in data and "trie" not in data


def test_load_missing_file_returns_empty_profile(tmp_path):
    assert UserProfile.load(tmp_path / "nope.json").token_count == 0


def test_save_load_preserves_devanagari(tmp_path):
    profile = make()
    profile.observe(["भारत", "एक", "देश"])
    restored = UserProfile.load(profile.save(tmp_path / "hi.json"))
    assert "भारत" in restored.trie


# --------------------------------------------------------------------- misc

def test_stats_reports_the_essentials():
    profile = make()
    profile.observe(["a", "b", "c"])
    stats = profile.stats()
    assert stats["tokens"] == 3 and stats["sentences"] == 1
    assert stats["vocabulary"] == 3


def test_repr_mentions_tokens_and_lambda():
    text = repr(make())
    assert "tokens=" in text and "lambda=" in text
