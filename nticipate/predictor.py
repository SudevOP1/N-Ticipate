"""
Phases 3 & 6: unified prediction API.

This is the single function the desktop app (Phase 7) calls. It hides
whether we're doing prefix completion or next-word prediction, and
whether POS reranking is on.
"""

import math

from nticipate.config import load_config
from nticipate.ngram import NgramModel
from nticipate.trie import Trie
from nticipate.hmm import HMMTagger
from nticipate.userprofile import UserProfile
from nticipate.preprocess import START_TOKEN, END_TOKEN, UNK_TOKEN, apply_truecase

CFG = load_config()

_SPECIAL_TOKENS = {START_TOKEN, END_TOKEN, UNK_TOKEN}
_LOG_FLOOR_PROB = 1e-12  # avoids math.log(0) for a zero-probability candidate


class Predictor:
    def __init__(
        self,
        ngram_model: NgramModel,
        trie: Trie,
        tagger: HMMTagger | None = None,
        user_profile: UserProfile | None = None,
        truecase_map: dict[str, str] | None = None,
    ):
        self.ngram_model = ngram_model
        self.trie = trie
        self.tagger = tagger
        self.user_profile = user_profile
        self.truecase_map = truecase_map or {}
        self.rerank_enabled = CFG["reranker"]["enabled"] and tagger is not None
        self.alpha = CFG["reranker"]["alpha"]

    def predict(self, context: tuple[str, ...], prefix: str = "", k: int | None = None) -> list[str]:
        """
        context: previous words already committed (e.g. last two words)
        prefix:  partial word currently being typed, "" if at a word boundary
        Returns up to k ranked completion/next-word strings, in natural
        (truecased) surface form if a truecase_map was provided.
        """
        k = k or CFG["predictor"]["top_k"]
        context = tuple(context)
        counting_prefix = prefix.lower() if CFG["preprocessing"]["lowercase_for_counts"] else prefix
        personalization_on = self.user_profile is not None and CFG["predictor"]["personalization"]["enabled"]

        # Candidates must be gathered from BOTH the base model and the
        # user model *before* scoring -- a word that only exists in the
        # user's own vocabulary (a name, jargon) can never be suggested
        # if it's never even added to the candidate set, no matter how
        # high its blended probability would be.
        candidates: set[str] = set()

        if counting_prefix:
            candidates.update(self.trie.words_with_prefix(counting_prefix))
            if personalization_on:
                candidates.update(
                    w for w in self.user_profile.vocab if w.lower().startswith(counting_prefix)
                )
        else:
            pool_size = max(k * 4, 20)
            candidates.update(w for w, _ in self.ngram_model.top_k(context, k=pool_size))
            if personalization_on:
                order = self.ngram_model.n
                candidates.update(w for w, _ in self.user_profile.top_k(context, k=pool_size, n=order))

        candidates -= _SPECIAL_TOKENS
        if not candidates:
            return []

        base_scores = {w: self.ngram_model.prob(w, context) for w in candidates}
        scores = self._blend_with_user_model(base_scores, context)

        if self.rerank_enabled:
            scores = self._rerank_with_pos(context, scores)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]
        words = [w for w, _ in ranked]

        if self.truecase_map:
            words = apply_truecase(words, self.truecase_map)

        return words

    def _blend_with_user_model(
        self, base_scores: dict[str, float], context: tuple[str, ...]
    ) -> dict[str, float]:
        """Interpolate base n-gram scores with the personal user model.
        score(w) = (1 - lambda) * P_base(w|context) + lambda * P_user(w|context)
        """
        if self.user_profile is None or not CFG["predictor"]["personalization"]["enabled"]:
            return base_scores

        lam = self.user_profile.current_lambda()
        if lam <= 0:
            return base_scores

        order = self.ngram_model.n
        return {
            word: (1 - lam) * base_p + lam * self.user_profile.prob(word, context, n=order)
            for word, base_p in base_scores.items()
        }

    def _rerank_with_pos(self, context: tuple[str, ...], candidates: dict[str, float]) -> dict[str, float]:
        """score(w) = log P(w|context) + alpha * log P(tag(w) | tag_context)

        `context` here is the same tuple of previous *words* used for
        the n-gram lookup, not tags -- it's tagged via the HMM's own
        viterbi() to get a tag_context for the transition lookup.
        Special tokens (<s>, </s>, <UNK>) are stripped first since the
        HMM was never trained on them and would otherwise fall through
        to a meaningless suffix-heuristic guess.

        The returned values are log-space combined scores, not valid
        probabilities -- fine, since predict() only uses them for
        ranking (sorting), never displays them.
        """
        context_words = [w for w in context if w not in _SPECIAL_TOKENS]
        context_tags = self.tagger.viterbi(context_words) if context_words else []
        tag_context = tuple(context_tags[-1:])  # bigram tag model: only the last tag matters

        next_tag_probs = self.tagger.next_tag_distribution(tag_context)

        reranked = {}
        for word, prob in candidates.items():
            word_tag = self.tagger.most_likely_tag_for_word(word)
            tag_prob = next_tag_probs.get(word_tag, _LOG_FLOOR_PROB)
            log_score = math.log(max(prob, _LOG_FLOOR_PROB))
            log_tag = math.log(max(tag_prob, _LOG_FLOOR_PROB))
            reranked[word] = log_score + self.alpha * log_tag
        return reranked
