"""Phase 3 tests — the prediction engine — and Phase 6, the POS reranker.

Phase 6 adds no new class: it is a term inside :meth:`Predictor.predict`, so
its tests belong next to the ranking they change.
"""

from __future__ import annotations

import numpy as np
import pytest

from nticipate.hmm import HMMTagger
from nticipate.ngram import NgramModel
from nticipate.predictor import (
    Mode,
    Predictor,
    alpha_sweep,
    completion_hit_at_k,
    hit_at_k,
    keystroke_savings,
    latency_stats,
    rerank_ablation,
    reranking,
    tag_window_disagreement,
    typical_tag_agreement,
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


# ==========================================================================
# Phase 6 — POS-aware reranking
# ==========================================================================

# The same sentences as TRAIN, tagged. Keeping the two corpora aligned is what
# lets these tests say anything about ranking: a tagger that had never seen the
# candidate words would send every one of them down the unknown-word path and
# the POS term would be measuring the suffix heuristic instead.
TAGGED = [
    [("i", "PRON"), ("would", "VERB"), ("like", "VERB"), ("to", "PRT"),
     ("know", "VERB"), ("more", "ADJ")],
    [("i", "PRON"), ("would", "VERB"), ("like", "VERB"), ("to", "PRT"),
     ("know", "VERB"), ("why", "ADV")],
    [("i", "PRON"), ("would", "VERB"), ("like", "VERB"), ("to", "PRT"),
     ("thank", "VERB"), ("you", "PRON")],
    [("the", "DET"), ("recent", "ADJ"), ("report", "NOUN"), ("was", "VERB"),
     ("good", "ADJ")],
    [("the", "DET"), ("record", "NOUN"), ("was", "VERB"), ("broken", "VERB")],
    [("please", "ADV"), ("recommend", "VERB"), ("a", "DET"), ("good", "ADJ"),
     ("book", "NOUN")],
    [("the", "DET"), ("report", "NOUN"), ("was", "VERB"), ("good", "ADJ")],
] * 3


def make_tagger() -> HMMTagger:
    return HMMTagger().fit(TAGGED)


def make_reranking_predictor(**kw) -> Predictor:
    kw.setdefault("tagger", make_tagger())
    return make_predictor(**kw)


# ------------------------------------------------------------------- wiring

def test_no_tagger_means_no_reranking():
    predictor = make_predictor()
    assert predictor.tagger is None
    assert predictor.rerank_active is False
    assert all(s.tag is None for s in predictor.predict(["like", "to"]))


def test_attaching_a_tagger_turns_reranking_on():
    predictor = make_predictor()
    predictor.attach_tagger(make_tagger())
    assert predictor.rerank_active is True


def test_rerank_flag_disables_the_term_even_with_a_tagger():
    predictor = make_reranking_predictor(rerank=False)
    assert predictor.tagger is not None
    assert predictor.rerank_active is False


def test_unfitted_tagger_is_not_active():
    predictor = make_predictor(tagger=HMMTagger())
    assert predictor.rerank_active is False


def test_attach_tagger_clears_the_caches():
    predictor = make_reranking_predictor()
    predictor.predict(["like", "to"])
    assert predictor._word_tag_cache
    predictor.attach_tagger(make_tagger())
    assert predictor._word_tag_cache == {}
    assert predictor._context_tag_cache == {}


def test_repr_reports_the_rerank_weight():
    assert "rerank=alpha=" in repr(make_reranking_predictor())
    assert "rerank=off" in repr(make_predictor())


def test_suggestions_carry_their_tag():
    for suggestion in make_reranking_predictor().predict(["like", "to"]):
        assert suggestion.tag in make_tagger().tags


# -------------------------------------------------------------- the tag term

def test_context_tag_is_none_without_context():
    assert make_reranking_predictor().context_tag([]) is None


def test_context_tag_matches_the_last_trellis_column():
    predictor = make_reranking_predictor()
    context = ["the", "report"]
    column = predictor.tagger.trellis(context).column(-1)
    assert predictor.context_tag(context) == next(iter(column))


def test_context_tag_does_not_charge_the_end_of_sentence_transition():
    # A buffer the user is still typing has not ended, so the transition into
    # the end-of-sentence state must not be charged to its last word. That
    # term is not small: the end state is reached overwhelmingly from
    # punctuation, and on the shipped English tagger it was enough to make
    # "i would" come back tagged PRON "." -- the tagger preferring to believe
    # "would" was a full stop over believing the sentence continued.
    #
    # Forcing the end distribution to a single tag makes the invariant exact:
    # viterbi() has to follow it, and context_tag() has to ignore it.
    predictor = make_reranking_predictor()
    tagger = predictor.tagger
    assert predictor.context_tag(["the", "report"]) == "NOUN"

    tagger.log_final = np.full(len(tagger.tags), -50.0)
    tagger.log_final[tagger.tags.index("ADV")] = 0.0
    predictor._context_tag_cache = {}

    assert tagger.viterbi(["the", "report"])[-1] == "ADV"
    assert predictor.context_tag(["the", "report"]) == "NOUN"


def test_context_tag_only_looks_at_the_window():
    predictor = make_reranking_predictor(tag_context_size=1)
    # With a one-token window only the last token can matter.
    assert predictor.context_tag(["the", "report"]) == predictor.context_tag(["report"])


def test_context_tag_is_cached():
    predictor = make_reranking_predictor()
    predictor.context_tag(["the", "report"])
    assert ("the", "report") in predictor._context_tag_cache


def test_typical_tag_of_a_known_word():
    predictor = make_reranking_predictor()
    assert predictor.typical_tag("report") == "NOUN"
    assert predictor.typical_tag("know") == "VERB"
    assert predictor.typical_tag("the") == "DET"


def test_typical_tag_of_an_unknown_word_uses_the_suffix_heuristic():
    # Never in TAGGED, so this is the Phase 4 unknown-word path doing the work.
    predictor = make_reranking_predictor()
    assert not predictor.tagger.knows("reconsidering")
    assert predictor.typical_tag("reconsidering") == "VERB"


def test_tag_score_is_the_transition_probability():
    predictor = make_reranking_predictor()
    tagger = predictor.tagger
    expected = float(
        tagger.log_transition[tagger.tags.index("DET"), tagger.tags.index("NOUN")]
    )
    assert predictor.tag_score("report", "NOUN", "DET") == pytest.approx(expected)


def test_tag_score_without_context_uses_the_initial_distribution():
    predictor = make_reranking_predictor()
    tagger = predictor.tagger
    expected = float(tagger.log_initial[tagger.tags.index("DET")])
    assert predictor.tag_score("the", "DET", None) == pytest.approx(expected)


def test_unknown_words_pay_the_penalty_exactly_once():
    predictor = make_reranking_predictor(unknown_tag_penalty=-2.0)
    known = predictor.tag_score("report", "NOUN", "DET")
    unknown = predictor.tag_score("zzzblort", "NOUN", "DET")
    assert unknown == pytest.approx(known - 2.0)


# ------------------------------------------------------------------ ranking

def test_alpha_zero_reproduces_the_phase_3_ranking_exactly():
    # The logarithm is monotonic, so a zero weight on the tag term cannot
    # reorder anything. This is the Phase 6 "done when" condition.
    plain = make_predictor()
    reranked = make_reranking_predictor(rerank_alpha=0.0)
    for context in ([], ["the"], ["like", "to"], ["was"]):
        assert [s.word for s in reranked.predict(context, k=5)] == [
            s.word for s in plain.predict(context, k=5)
        ]
    for prefix in ("re", "rec", "b"):
        assert [s.word for s in reranked.predict(["the"], prefix, k=5)] == [
            s.word for s in plain.predict(["the"], prefix, k=5)
        ]


def test_reranked_scores_are_log_scale():
    scores = [s.score for s in make_reranking_predictor().predict(["the"], k=5)]
    assert scores and all(score < 0 for score in scores)
    # ... and still sorted best-first.
    assert scores == sorted(scores, reverse=True)


def test_zero_probability_candidates_land_on_the_score_floor():
    # Only MLE ever hands the reranker a zero: stupid backoff floors an unseen
    # unigram on purpose (see NgramModel._stupid_backoff) precisely so that
    # candidates do not drop out of the ranking. "the book" is an unseen
    # bigram in TRAIN, so MLE scores it 0.0 and log(0) would be -inf.
    model = NgramModel(order=2, smoothing="mle").fit(TRAIN)
    predictor = Predictor(
        model, tagger=make_tagger(), personalization=False, rerank_alpha=0.0
    )
    assert model.prob("book", ["the"]) == 0.0
    hit = [s for s in predictor.predict(["the"], "b", k=5) if s.word == "book"]
    assert hit and hit[0].score == pytest.approx(predictor.score_floor)


def test_a_large_alpha_is_dominated_by_the_tag_term():
    # This is the mechanism the whole phase is built on, in one assertion:
    # after a determiner the tag bigram says NOUN or ADJ, so with the POS term
    # turned up the verbs and pronouns the base model was happy to offer are
    # gone, whatever their n-gram score.
    predictor = make_reranking_predictor(rerank_alpha=50.0)
    tags = {s.tag for s in predictor.predict(["the"], k=5)}
    assert tags <= {"NOUN", "ADJ"}

    plain = make_predictor()
    assert {s.word for s in plain.predict(["the"], k=5)} - {
        s.word for s in predictor.predict(["the"], k=5)
    }


def test_reranking_can_change_the_order():
    # Not a claim that it improves anything — only that the term is live. If
    # this ever passed vacuously the ablation below would be measuring noise.
    plain = make_predictor()
    reranked = make_reranking_predictor(rerank_alpha=5.0)
    orders = [
        ([s.word for s in plain.predict(c, k=5)],
         [s.word for s in reranked.predict(c, k=5)])
        for c in ([], ["the"], ["was"], ["like", "to"], ["a"])
    ]
    assert any(before != after for before, after in orders)


# ---------------------------------------------------------------- ablation

def test_reranking_context_manager_restores_state():
    predictor = make_reranking_predictor(rerank_alpha=0.3)
    with reranking(predictor, False, 0.9) as inner:
        assert inner.rerank is False
        assert inner.rerank_alpha == 0.9
    assert predictor.rerank is True
    assert predictor.rerank_alpha == 0.3


def test_reranking_context_manager_restores_state_on_error():
    predictor = make_reranking_predictor()
    with pytest.raises(RuntimeError):
        with reranking(predictor, False):
            raise RuntimeError("boom")
    assert predictor.rerank is True


def test_rerank_ablation_reports_both_arms_and_the_delta():
    predictor = make_reranking_predictor()
    result = rerank_ablation(predictor, TRAIN[:4], ks=(1, 3), limit=4)
    for arm in ("off", "on"):
        for task in ("next_word", "completion"):
            assert "hit@1" in result[arm][task] and "hit@3" in result[arm][task]
    for task in ("next_word", "completion"):
        assert result["delta"][task]["hit@1"] == pytest.approx(
            result["on"][task]["hit@1"] - result["off"][task]["hit@1"]
        )


def test_rerank_ablation_needs_a_tagger():
    with pytest.raises(ValueError):
        rerank_ablation(make_predictor(), TRAIN[:2])


def test_ablation_leaves_the_predictor_as_it_found_it():
    predictor = make_reranking_predictor(rerank_alpha=0.3)
    rerank_ablation(predictor, TRAIN[:2], ks=(1,), limit=2, alpha=0.9)
    assert predictor.rerank is True and predictor.rerank_alpha == 0.3


def test_alpha_zero_row_of_the_sweep_matches_reranking_off():
    predictor = make_reranking_predictor()
    rows = alpha_sweep(predictor, TRAIN[:4], alphas=(0.0, 0.5), ks=(1, 3), limit=4)
    with reranking(predictor, False):
        baseline = hit_at_k(predictor, TRAIN[:4], ks=(1, 3), limit=4)
    assert rows[0]["hit@1"] == pytest.approx(baseline["hit@1"])
    assert rows[0]["hit@3"] == pytest.approx(baseline["hit@3"])


def test_alpha_sweep_can_measure_completion_too():
    predictor = make_reranking_predictor()
    rows = alpha_sweep(
        predictor, TRAIN[:4], alphas=(0.0, 0.3), ks=(1,), limit=4,
        mode=Mode.COMPLETION,
    )
    assert [row["mode"] for row in rows] == ["completion", "completion"]
    assert all("prefix_len" in row for row in rows)


def test_tag_window_disagreement_separates_window_from_right_context():
    predictor = make_reranking_predictor()
    result = tag_window_disagreement(predictor, TRAIN[:5])
    assert result["positions"] > 0
    assert 0.0 <= result["vs_prefix"] <= 1.0
    assert 0.0 <= result["vs_sentence"] <= 1.0
    assert result["window"] == predictor.tag_context_size


def test_a_window_as_long_as_the_sentence_never_disagrees_with_the_prefix():
    # Sanity check that vs_prefix is measuring the window and nothing else.
    predictor = make_reranking_predictor(tag_context_size=100)
    assert tag_window_disagreement(predictor, TRAIN[:5])["vs_prefix"] == 0.0


def test_typical_tag_agreement_is_a_rate():
    predictor = make_reranking_predictor()
    result = typical_tag_agreement(predictor, TRAIN[:5])
    assert result["tokens"] > 0
    assert 0.0 <= result["agreement"] <= 1.0
    assert all(len(pair) == 2 for pair, _ in result["top_confusions"])


# ------------------------------------------------------------- persistence

def test_from_paths_loads_the_tagger(tmp_path):
    model = NgramModel(order=3, smoothing="stupid_backoff").fit(TRAIN)
    model_path = model.save(tmp_path / "model.pkl")
    tagger_path = make_tagger().save(tmp_path / "tagger.pkl")

    predictor = Predictor.from_paths(model_path, tagger_path=tagger_path)
    assert predictor.rerank_active is True
    assert predictor.typical_tag("report") == "NOUN"


def test_from_paths_reads_a_truecase_only_artefact(tmp_path):
    from nticipate.preprocess import preprocess_corpus

    corpus = preprocess_corpus("India is here. India is there. India again.",
                               min_sentence_tokens=1)
    truecase_path = corpus.save_truecase(tmp_path / "truecase.json")
    model = NgramModel(order=3, smoothing="stupid_backoff").fit(TRAIN)
    model_path = model.save(tmp_path / "model.pkl")

    predictor = Predictor.from_paths(model_path, truecase_path=truecase_path)
    assert predictor.truecase == corpus.truecase


def test_from_paths_prefers_the_truecase_artefact_over_the_corpus(tmp_path):
    from nticipate.preprocess import Corpus, Vocab

    corpus = Corpus(train=[], dev=[], test=[], vocab=Vocab({"x"}),
                    truecase={"x": "CORPUS"})
    corpus_path = corpus.save(tmp_path / "corpus.json")
    truecase_path = tmp_path / "truecase.json"
    truecase_path.write_text('{"x": "ARTEFACT"}', encoding="utf-8")
    model_path = NgramModel(order=3, smoothing="stupid_backoff").fit(TRAIN).save(
        tmp_path / "model.pkl")

    predictor = Predictor.from_paths(model_path, corpus_path=corpus_path,
                                     truecase_path=truecase_path)
    assert predictor.truecase == {"x": "ARTEFACT"}


def test_reranking_survives_a_tagger_round_trip(tmp_path):
    path = make_tagger().save(tmp_path / "tagger.pkl")
    loaded = make_reranking_predictor(tagger=HMMTagger.load(path))
    fresh = make_reranking_predictor()
    assert [s.word for s in loaded.predict(["the"], k=5)] == [
        s.word for s in fresh.predict(["the"], k=5)
    ]


# --------------------------------------------------------------- latency

def test_reranking_stays_inside_the_debounce_budget():
    # The whole point of a context-free candidate tag: reranking must not turn
    # one decode per keystroke into one per candidate.
    predictor = make_reranking_predictor()
    contexts = [(["like", "to"], ""), (["the"], "re"), ([], "b")]
    stats = latency_stats(predictor, contexts, repeats=20, percentiles=(95,))
    assert stats["p95_ms"] < 50
