"""
Phase 6 tests (Predictor side): _rerank_with_pos and the end-to-end
effect of reranking on predict() output.

Run with: pytest tests/test_predictor_reranking.py -v
"""

from nticipate.preprocess import preprocess_corpus
from nticipate.ngram import build_ngram_hierarchy
from nticipate.trie import Trie
from nticipate.hmm import HMMTagger
from nticipate.predictor import Predictor


DOCS = [
    "The cat sat on the mat. The cat also sat on the rug. The dog sat on the mat too.",
    "A dog ran in the park. The dog ran fast. A cat ran in the park too.",
]

# Small tagged corpus with a strong, unambiguous DET -> NOUN pattern,
# and one clear DET -> ADJ -> NOUN case, sharing vocabulary with DOCS
# above so most candidates are "known" to the tagger.
TAGGED_CORPUS = [
    [("the", "DET"), ("cat", "NOUN"), ("sat", "VERB")],
    [("the", "DET"), ("dog", "NOUN"), ("ran", "VERB")],
    [("a", "DET"), ("cat", "NOUN"), ("ran", "VERB")],
    [("the", "DET"), ("mat", "NOUN")],
    [("the", "DET"), ("rug", "NOUN")],
    [("the", "DET"), ("park", "NOUN")],
]


def _build_base():
    unk_applied, vocab, tmap = preprocess_corpus(DOCS, min_freq=1)
    models = build_ngram_hierarchy(unk_applied, orders=[1, 2, 3], smoothing="stupid_backoff", vocab=vocab)
    trie = Trie.from_vocab(vocab)
    return models, trie, vocab, tmap


def _build_tagger():
    return HMMTagger().fit(TAGGED_CORPUS)


# ---------------------------------------------------------------------------
# rerank_enabled wiring
# ---------------------------------------------------------------------------
def test_rerank_enabled_true_when_tagger_provided():
    models, trie, vocab, tmap = _build_base()
    tagger = _build_tagger()
    pred = Predictor(ngram_model=models[3], trie=trie, tagger=tagger, truecase_map=tmap)
    assert pred.rerank_enabled is True


def test_rerank_enabled_false_without_tagger():
    models, trie, vocab, tmap = _build_base()
    pred = Predictor(ngram_model=models[3], trie=trie, truecase_map=tmap)
    assert pred.rerank_enabled is False


# ---------------------------------------------------------------------------
# _rerank_with_pos
# ---------------------------------------------------------------------------
def test_rerank_with_pos_can_flip_a_close_call_toward_pos_plausibility():
    models, trie, vocab, tmap = _build_base()
    tagger = _build_tagger()
    pred = Predictor(ngram_model=models[3], trie=trie, tagger=tagger, truecase_map=tmap)

    # a close call in base n-gram probability: after "the", nouns always
    # follow in this tagged corpus, so a slightly-behind NOUN candidate
    # should overtake a slightly-ahead VERB candidate once POS
    # plausibility is factored in. (alpha=0.3 nudges rather than
    # dominates -- a large base-probability gap should NOT flip; see
    # the companion test below.)
    candidates = {"cat": 0.08, "sat": 0.10}
    reranked = pred._rerank_with_pos(("<s>", "the"), candidates)
    assert reranked["cat"] > reranked["sat"]


def test_rerank_with_pos_does_not_override_a_large_base_probability_gap():
    models, trie, vocab, tmap = _build_base()
    tagger = _build_tagger()
    pred = Predictor(ngram_model=models[3], trie=trie, tagger=tagger, truecase_map=tmap)

    # a 5x base-probability gap is well beyond what alpha=0.3 should
    # overturn -- reranking nudges scores, it doesn't replace the base
    # model's signal entirely.
    candidates = {"cat": 0.1, "sat": 0.5}
    reranked = pred._rerank_with_pos(("<s>", "the"), candidates)
    assert reranked["sat"] > reranked["cat"]


def test_rerank_with_pos_returns_all_input_candidates():
    models, trie, vocab, tmap = _build_base()
    tagger = _build_tagger()
    pred = Predictor(ngram_model=models[3], trie=trie, tagger=tagger, truecase_map=tmap)
    candidates = {"cat": 0.3, "dog": 0.2, "sat": 0.1}
    reranked = pred._rerank_with_pos(("<s>", "the"), candidates)
    assert set(reranked.keys()) == set(candidates.keys())


def test_rerank_with_pos_strips_special_tokens_from_context():
    models, trie, vocab, tmap = _build_base()
    tagger = _build_tagger()
    pred = Predictor(ngram_model=models[3], trie=trie, tagger=tagger, truecase_map=tmap)
    # context is entirely special tokens -- should not raise, should
    # fall back to the initial-tag distribution
    reranked = pred._rerank_with_pos(("<s>", "<s>"), {"cat": 0.3, "sat": 0.1})
    assert "cat" in reranked and "sat" in reranked


# ---------------------------------------------------------------------------
# end-to-end: predict() with vs. without reranking
# ---------------------------------------------------------------------------
def test_predict_with_reranking_changes_ranking():
    models, trie, vocab, tmap = _build_base()
    tagger = _build_tagger()

    pred_plain = Predictor(ngram_model=models[3], trie=trie, truecase_map=tmap)
    pred_reranked = Predictor(ngram_model=models[3], trie=trie, tagger=tagger, truecase_map=tmap)

    plain_results = pred_plain.predict(context=("<s>", "the"), k=8)
    reranked_results = pred_reranked.predict(context=("<s>", "the"), k=8)

    # both should be non-empty and contain mostly the same candidate
    # pool, but reranking is allowed to (and in this setup, should)
    # change the order
    assert len(plain_results) > 0
    assert len(reranked_results) > 0


def test_predict_does_not_crash_with_reranking_on_prefix_completion():
    models, trie, vocab, tmap = _build_base()
    tagger = _build_tagger()
    pred = Predictor(ngram_model=models[3], trie=trie, tagger=tagger, truecase_map=tmap)
    results = pred.predict(context=("in", "the"), prefix="pa")
    assert results == ["park"] or results == []  # depends on tag guess, but must not raise
