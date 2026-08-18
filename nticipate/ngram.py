"""
Phase 2: n-gram language model.

Implements, per CFG['ngram']['smoothing']:
    - mle              raw maximum likelihood (for demonstrating zero-prob issue)
    - laplace          add-k smoothing
    - stupid_backoff    fast, what the running app actually uses
    - kneser_ney        interpolated KN, the quality ceiling for comparison

Models of different orders are chained via `lower_order_model` (built by
build_ngram_hierarchy below) so stupid_backoff and interpolated
Kneser-Ney can recurse down to lower orders -- e.g. trigram backs off
to bigram backs off to unigram.

See notebooks/02_ngram_models.ipynb for the perplexity sweep across
n in {1,2,3} and smoothing method.
"""

import math
import pickle
import random
from collections import defaultdict, Counter

from nticipate.config import load_config
from nticipate.preprocess import pad_sentence, START_TOKEN, END_TOKEN

CFG = load_config()

VALID_SMOOTHING = {"mle", "laplace", "stupid_backoff", "kneser_ney"}


class NgramModel:
    def __init__(self, n: int, smoothing: str | None = None, lower_order_model: "NgramModel | None" = None):
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = n
        self.smoothing = smoothing or CFG["ngram"]["smoothing"]
        if self.smoothing not in VALID_SMOOTHING:
            raise ValueError(f"unknown smoothing method: {self.smoothing!r}, expected one of {VALID_SMOOTHING}")
        self.lower_order_model = lower_order_model  # NgramModel of order n-1, or None

        self.counts: dict[tuple, Counter] = defaultdict(Counter)   # context -> Counter(word)
        self.context_totals: Counter = Counter()                    # context -> total count
        self.vocab: set[str] = set()
        self.vocab_size: int = 0

        # Kneser-Ney bookkeeping (built lazily by fit() when smoothing == "kneser_ney")
        self._kn_continuation_counts: Counter | None = None   # word -> # distinct contexts preceding it
        self._kn_total_types: int | None = None               # total distinct (context, word) pairs
        self._kn_unigram_continuation: Counter | None = None  # installed from the bigram model, unigram only
        self._kn_unigram_total_types: int | None = None

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    def fit(self, tokenized_sentences: list[list[str]], vocab: set[str] | None = None) -> "NgramModel":
        """Count n-grams from unpadded, <UNK>-applied sentences. Padding is
        order-specific, so it happens here (via preprocess.pad_sentence),
        not in preprocess.py -- see that module's docstring.
        """
        self.counts = defaultdict(Counter)
        self.context_totals = Counter()
        observed_vocab: set[str] = set()

        for sentence in tokenized_sentences:
            padded = pad_sentence(sentence, self.n)
            observed_vocab.update(padded)
            for i in range(self.n - 1, len(padded)):
                context = tuple(padded[i - self.n + 1 : i])
                word = padded[i]
                self.counts[context][word] += 1
                self.context_totals[context] += 1

        # Prefer the shared preprocessing vocab (keeps Laplace's V consistent
        # across train/dev/test) but fall back to what we observed if none given.
        self.vocab = vocab if vocab is not None else observed_vocab
        self.vocab_size = len(self.vocab)

        if self.smoothing == "kneser_ney":
            self._build_kn_structures()

        return self

    def _build_kn_structures(self) -> None:
        continuation_counts: Counter = Counter()
        for word_counts in self.counts.values():
            for word in word_counts:
                continuation_counts[word] += 1
        self._kn_continuation_counts = continuation_counts
        self._kn_total_types = sum(len(wc) for wc in self.counts.values())

    def _install_kn_continuation_base(self, bigram_model: "NgramModel") -> None:
        """For the n==1 model only: use continuation probability derived
        from the bigram model's counts (Chen & Goodman) instead of raw
        unigram frequency. Called by build_ngram_hierarchy.
        """
        self._kn_unigram_continuation = bigram_model._kn_continuation_counts
        self._kn_unigram_total_types = bigram_model._kn_total_types

    # ------------------------------------------------------------------
    # context helpers (read-only -- never mutate self.counts via [])
    # ------------------------------------------------------------------
    def _normalize_context(self, context: tuple[str, ...]) -> tuple[str, ...]:
        need = self.n - 1
        context = tuple(context)
        if need == 0:
            return ()
        if len(context) < need:
            return (START_TOKEN,) * (need - len(context)) + context
        if len(context) > need:
            return context[-need:]
        return context

    def _context_word_count(self, context: tuple[str, ...], word: str) -> int:
        return self.counts.get(context, {}).get(word, 0)

    def _context_total(self, context: tuple[str, ...]) -> int:
        return self.context_totals.get(context, 0)

    @staticmethod
    def _drop_oldest(context: tuple[str, ...]) -> tuple[str, ...]:
        """Drop the leftmost (oldest) word -- how you back off from an
        n-gram context to an (n-1)-gram context."""
        return context[1:] if context else ()

    # ------------------------------------------------------------------
    # probability
    # ------------------------------------------------------------------
    def prob(self, word: str, context: tuple[str, ...]) -> float:
        """P(word | context) under self.smoothing."""
        context = self._normalize_context(context)
        if self.smoothing == "mle":
            return self._prob_mle(word, context)
        if self.smoothing == "laplace":
            return self._prob_laplace(word, context)
        if self.smoothing == "stupid_backoff":
            return self._prob_stupid_backoff(word, context)
        if self.smoothing == "kneser_ney":
            return self._prob_kneser_ney(word, context)
        raise ValueError(f"unknown smoothing method: {self.smoothing!r}")

    def _prob_mle(self, word: str, context: tuple[str, ...]) -> float:
        total = self._context_total(context)
        if total == 0:
            return 0.0
        return self._context_word_count(context, word) / total

    def _prob_laplace(self, word: str, context: tuple[str, ...]) -> float:
        k = CFG["ngram"]["add_k"]
        total = self._context_total(context)
        count = self._context_word_count(context, word)
        v = max(self.vocab_size, 1)
        return (count + k) / (total + k * v)

    def _prob_stupid_backoff(self, word: str, context: tuple[str, ...]) -> float:
        count = self._context_word_count(context, word)
        total = self._context_total(context)
        if total > 0 and count > 0:
            return count / total

        if self.n == 1 or self.lower_order_model is None:
            # base case: no lower order to fall back to. Floor with
            # add-one smoothing over the vocabulary so truly unseen
            # words get a small non-zero probability instead of exactly 0.
            unigram_total = sum(self.context_totals.values())
            return 1.0 / (unigram_total + max(self.vocab_size, 1))

        alpha = CFG["ngram"]["backoff_alpha"]
        lower_context = self._drop_oldest(context)
        return alpha * self.lower_order_model._prob_stupid_backoff(word, lower_context)

    def _prob_kneser_ney(self, word: str, context: tuple[str, ...]) -> float:
        d = CFG["ngram"]["kn_discount"]

        if self.n == 1:
            if self._kn_unigram_continuation is not None and self._kn_unigram_total_types:
                cont = self._kn_unigram_continuation.get(word, 0)
                return cont / self._kn_unigram_total_types
            # fallback if no bigram model was available to derive
            # continuation counts from: add-one-smoothed raw unigram MLE
            total = sum(self.context_totals.values())
            count = self._context_word_count((), word)
            return (count + 1) / (total + max(self.vocab_size, 1))

        total = self._context_total(context)
        lower_context = self._drop_oldest(context)
        lower_prob = (
            self.lower_order_model._prob_kneser_ney(word, lower_context)
            if self.lower_order_model is not None
            else 0.0
        )
        if total == 0:
            # unseen context entirely -- fall back completely to lower order
            return lower_prob

        count = self._context_word_count(context, word)
        discounted = max(count - d, 0) / total
        num_distinct_following = len(self.counts.get(context, {}))
        lam = (d * num_distinct_following) / total
        return discounted + lam * lower_prob

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------
    def top_k(self, context: tuple[str, ...], k: int = 5) -> list[tuple[str, float]]:
        """Top-k (word, prob) continuations for a context, used by predictor.py."""
        context = self._normalize_context(context)
        candidates: set[str] = set()

        if context in self.counts:
            candidates.update(self.counts[context].keys())

        # Smoothing methods that backoff/interpolate can still produce a
        # sensible answer for an unseen context -- widen the candidate
        # pool from the lower-order model rather than returning nothing.
        if self.smoothing in ("stupid_backoff", "kneser_ney") and self.lower_order_model is not None:
            lower_context = self._drop_oldest(context)
            candidates.update(w for w, _ in self.lower_order_model.top_k(lower_context, k=k * 3))

        candidates.discard(START_TOKEN)
        scored = [(w, self.prob(w, context)) for w in candidates]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]

    # ------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------
    def perplexity(self, tokenized_sentences: list[list[str]]) -> float:
        """Perplexity on held-out (dev/test) data. Core Phase 2 metric.

        MLE will frequently assign exact zero probability to unseen
        n-grams on held-out data -- that's floored to a tiny epsilon
        here (rather than raising) so perplexity comes back as a very
        large finite number instead of crashing. That huge number *is*
        the demonstration of MLE's sparsity problem; report it as such
        rather than treating it as a bug.
        """
        epsilon = 1e-12
        log_prob_sum = 0.0
        token_count = 0

        for sentence in tokenized_sentences:
            padded = pad_sentence(sentence, self.n)
            for i in range(self.n - 1, len(padded)):
                context = tuple(padded[i - self.n + 1 : i])
                word = padded[i]
                p = self.prob(word, context)
                if p <= 0:
                    p = epsilon
                log_prob_sum += math.log(p)
                token_count += 1

        if token_count == 0:
            return float("inf")
        avg_log_prob = log_prob_sum / token_count
        return math.exp(-avg_log_prob)

    # ------------------------------------------------------------------
    # packaging
    # ------------------------------------------------------------------
    def prune(self, min_count: int | None = None, top_k_per_context: int | None = None) -> "NgramModel":
        """Drop rare n-grams and cap continuations per context (for app
        packaging). Mutates self in place and returns self for chaining.

        Note: after pruning, context_totals reflects only the *kept*
        counts, so probabilities within a context still sum to ~1 over
        the surviving words -- but the pruned model no longer represents
        true corpus frequencies. Expect (and report) higher perplexity
        on the pruned model vs. the full model -- that gap is the
        size/quality trade-off the app is making.
        """
        min_count = min_count if min_count is not None else CFG["ngram"]["prune_min_count"]
        top_k_per_context = (
            top_k_per_context if top_k_per_context is not None else CFG["ngram"]["prune_top_k_per_context"]
        )

        new_counts: dict[tuple, Counter] = defaultdict(Counter)
        new_totals: Counter = Counter()

        for context, word_counts in self.counts.items():
            filtered = {w: c for w, c in word_counts.items() if c >= min_count}
            if not filtered:
                continue
            top_items = sorted(filtered.items(), key=lambda item: item[1], reverse=True)[:top_k_per_context]
            for word, count in top_items:
                new_counts[context][word] = count
                new_totals[context] += count

        self.counts = new_counts
        self.context_totals = new_totals
        return self

    def model_size_estimate(self) -> dict:
        """Rough size stats for the report: number of distinct contexts,
        number of (context, word) count entries, and an approximate
        serialized size in bytes -- useful for a before/after-pruning
        comparison table.
        """
        return {
            "num_contexts": len(self.counts),
            "num_entries": sum(len(wc) for wc in self.counts.values()),
            "size_bytes": len(pickle.dumps(self.counts)),
        }

    # ------------------------------------------------------------------
    # generation (qualitative demo)
    # ------------------------------------------------------------------
    def generate(self, max_len: int = 20, rng: random.Random | None = None) -> list[str]:
        """Sample a sentence -- useful qualitative demo of model order/smoothing.
        Unigram output tends to be word salad; trigram output is closer
        to grammatical. That contrast is the point of this method.
        """
        rng = rng or random.Random()
        candidates = sorted(w for w in self.vocab if w != START_TOKEN)
        context = (START_TOKEN,) * (self.n - 1)
        tokens: list[str] = []

        for _ in range(max_len):
            weights = [self.prob(w, context) for w in candidates]
            total_weight = sum(weights)
            if total_weight <= 0:
                break
            r = rng.uniform(0, total_weight)
            upto = 0.0
            chosen = candidates[-1]
            for word, weight in zip(candidates, weights):
                upto += weight
                if upto >= r:
                    chosen = word
                    break
            if chosen == END_TOKEN:
                break
            tokens.append(chosen)
            if self.n > 1:
                context = (context + (chosen,))[-(self.n - 1):]

        return tokens


def build_ngram_hierarchy(
    tokenized_sentences: list[list[str]],
    orders: list[int] | None = None,
    smoothing: str | None = None,
    vocab: set[str] | None = None,
) -> dict[int, NgramModel]:
    """Build one NgramModel per requested order, wiring each higher-order
    model's lower_order_model to the previous order so stupid_backoff and
    interpolated Kneser-Ney can recurse down. Returns {order: NgramModel}.

    This is the entry point notebooks/app code should use -- constructing
    NgramModel directly skips the lower_order_model wiring that backoff
    and Kneser-Ney depend on.
    """
    orders = sorted(orders or CFG["ngram"]["orders"])
    smoothing = smoothing or CFG["ngram"]["smoothing"]

    models: dict[int, NgramModel] = {}
    prev_model: NgramModel | None = None
    for n in orders:
        model = NgramModel(n=n, smoothing=smoothing, lower_order_model=prev_model)
        model.fit(tokenized_sentences, vocab=vocab)
        models[n] = model
        prev_model = model

    if smoothing == "kneser_ney" and 1 in models and 2 in models:
        models[1]._install_kn_continuation_base(models[2])

    return models
