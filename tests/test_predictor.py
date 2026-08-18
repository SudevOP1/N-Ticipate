"""
Phase 3 tests: unified Predictor (n-gram + trie + personalization).

Run with: pytest tests/test_predictor.py -v
"""

from nticipate.preprocess import preprocess_corpus, START_TOKEN, END_TOKEN, UNK_TOKEN
from nticipate.ngram import build_ngram_hierarchy
from nticipate.trie import Trie
from nticipate.predictor import Predictor
from nticipate.userprofile import UserProfile


def _build_base(docs, min_freq=1):
    unk_applied, vocab, truecase_map = preprocess_corpus(docs, min_freq=min_freq)
    models = build_ngram_hierarchy(unk_applied, orders=[1, 2, 3], smoothing="stupid_backoff", vocab=vocab)
    trie = Trie.from_vocab(vocab)
    return models, trie, vocab, truecase_map


DOCS = [
    "The cat sat on the mat. The cat also sat on the rug. The dog sat on the mat too.",
    "A dog ran in the park. The dog ran fast. A cat ran in the park too.",
]


# ---------------------------------------------------------------------------
# next-word prediction
# ---------------------------------------------------------------------------
def test_predict_next_word_returns_plausible_candidates():
    models, trie, vocab, tmap = _build_base(DOCS)
    pred = Predictor(ngram_model=models[3], trie=trie, truecase_map=tmap)
    results = pred.predict(context=("<s>", "the"), prefix="")
    assert len(results) > 0
    assert "cat" in results or "dog" in results


def test_predict_next_word_excludes_special_tokens():
    models, trie, vocab, tmap = _build_base(DOCS)
    pred = Predictor(ngram_model=models[3], trie=trie, truecase_map=tmap)
    results = pred.predict(context=("<s>", "the"), prefix="", k=20)
    assert START_TOKEN not in results
    assert END_TOKEN not in results
    assert UNK_TOKEN not in results


def test_predict_respects_k():
    models, trie, vocab, tmap = _build_base(DOCS)
    pred = Predictor(ngram_model=models[3], trie=trie, truecase_map=tmap)
    results = pred.predict(context=("<s>", "the"), prefix="", k=2)
    assert len(results) <= 2


# ---------------------------------------------------------------------------
# prefix completion
# ---------------------------------------------------------------------------
def test_predict_prefix_completion_matches_only_that_prefix():
    models, trie, vocab, tmap = _build_base(DOCS)
    pred = Predictor(ngram_model=models[3], trie=trie, truecase_map=tmap)
    results = pred.predict(context=("in", "the"), prefix="pa")
    assert results == ["park"]


def test_predict_prefix_completion_no_match_returns_empty():
    models, trie, vocab, tmap = _build_base(DOCS)
    pred = Predictor(ngram_model=models[3], trie=trie, truecase_map=tmap)
    assert pred.predict(context=("the",), prefix="zzz") == []


def test_predict_prefix_completion_is_case_insensitive():
    models, trie, vocab, tmap = _build_base(DOCS)
    pred = Predictor(ngram_model=models[3], trie=trie, truecase_map=tmap)
    lower = pred.predict(context=("in", "the"), prefix="pa")
    upper = pred.predict(context=("in", "the"), prefix="PA")
    assert lower == upper


# ---------------------------------------------------------------------------
# truecasing
# ---------------------------------------------------------------------------
def test_predict_restores_truecasing_when_map_provided():
    docs = ["India is a large country. India has many languages."]
    models, trie, vocab, tmap = _build_base(docs)
    pred = Predictor(ngram_model=models[2], trie=trie, truecase_map=tmap)
    results = pred.predict(context=("<s>",), prefix="in")
    # majority casing for "india" in this doc is capitalized (2 vs 0)
    assert "India" in results


def test_predict_without_truecase_map_returns_lowercased_form():
    docs = ["India is a large country. India has many languages."]
    models, trie, vocab, tmap = _build_base(docs)
    pred = Predictor(ngram_model=models[2], trie=trie)  # no truecase_map
    results = pred.predict(context=("<s>",), prefix="in")
    assert "india" in results


# ---------------------------------------------------------------------------
# personalization
# ---------------------------------------------------------------------------
def test_predict_surfaces_user_only_word_when_lambda_is_high():
    models, trie, vocab, tmap = _build_base(DOCS)
    profile = UserProfile()
    for _ in range(500):
        profile.observe_sentence(["the", "india", "trip", "was", "great"])

    pred_with = Predictor(ngram_model=models[3], trie=trie, user_profile=profile, truecase_map=tmap)
    pred_without = Predictor(ngram_model=models[3], trie=trie, truecase_map=tmap)

    with_results = pred_with.predict(context=("<s>", "the"), k=8)
    without_results = pred_without.predict(context=("<s>", "the"), k=8)

    assert "india" in with_results
    assert "india" not in without_results


def test_predict_prefix_completion_finds_user_only_word():
    models, trie, vocab, tmap = _build_base(DOCS)
    profile = UserProfile()
    for _ in range(50):
        profile.observe_sentence(["the", "india", "trip", "was", "great"])

    pred = Predictor(ngram_model=models[3], trie=trie, user_profile=profile, truecase_map=tmap)
    assert pred.predict(context=("the",), prefix="ind") == ["india"]

    pred_without = Predictor(ngram_model=models[3], trie=trie, truecase_map=tmap)
    assert pred_without.predict(context=("the",), prefix="ind") == []


def test_personalization_disabled_in_config_ignores_user_profile(monkeypatch):
    import nticipate.predictor as predictor_module

    models, trie, vocab, tmap = _build_base(DOCS)
    profile = UserProfile()
    for _ in range(500):
        profile.observe_sentence(["the", "india", "trip", "was", "great"])

    original = predictor_module.CFG["predictor"]["personalization"]["enabled"]
    predictor_module.CFG["predictor"]["personalization"]["enabled"] = False
    try:
        pred = Predictor(ngram_model=models[3], trie=trie, user_profile=profile, truecase_map=tmap)
        results = pred.predict(context=("<s>", "the"), k=8)
        assert "india" not in results
    finally:
        predictor_module.CFG["predictor"]["personalization"]["enabled"] = original


# ---------------------------------------------------------------------------
# rerank_enabled flag (should stay off when no tagger is provided,
# regardless of Phase 6's reranking now being implemented -- see
# test_predictor_reranking.py for the tagger-provided case)
# ---------------------------------------------------------------------------
def test_rerank_disabled_without_tagger():
    models, trie, vocab, tmap = _build_base(DOCS)
    pred = Predictor(ngram_model=models[3], trie=trie, truecase_map=tmap)
    assert pred.rerank_enabled is False
    # should not raise NotImplementedError since reranking is skipped
    pred.predict(context=("<s>", "the"), prefix="")
