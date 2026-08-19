"""Phase 2 — n-gram language model.

Counts unigrams, bigrams and trigrams (any order, really) and scores a word
given its left context under one of four smoothing methods:

============== =========================================================
``mle``        Maximum likelihood. The baseline that demonstrates the
               zero-probability problem: any unseen n-gram gets
               probability 0, so held-out perplexity is infinite.
``laplace``    Add-k. The classic textbook fix.
``stupid_backoff`` Fast, unnormalised backoff. What the running app uses,
               because it needs one dictionary lookup per level and no
               renormalisation inside a ~50 ms keystroke budget.
``kneser_ney`` Interpolated Kneser-Ney. The quality ceiling to compare
               against — it is the right model and the wrong speed.
============== =========================================================

The important structural point is that ``stupid_backoff`` is **not a
probability distribution** — it does not sum to one over the vocabulary. That
is fine for ranking candidates, which is all the app does with it, but it
means its "perplexity" is not comparable with the others on equal terms. The
model reports that honestly rather than quietly printing a number
(:attr:`NgramModel.is_normalized`).
"""

from __future__ import annotations

import math
import pickle
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from nticipate.config import get, resolve_path
from nticipate.preprocess import Sentence, pad_sentence

Context = tuple[str, ...]

SMOOTHING_METHODS = ("mle", "laplace", "stupid_backoff", "kneser_ney")

#: Methods whose scores form a real probability distribution over the
#: vocabulary. Perplexity is only strictly meaningful for these.
NORMALIZED_METHODS = ("mle", "laplace", "kneser_ney")


@dataclass
class ModelStats:
    order: int
    smoothing: str
    vocab_size: int
    ngram_counts: dict[int, int]
    total_ngrams: int


class NgramModel:
    """An n-gram language model with pluggable smoothing.

    ``order`` is the maximum n: 1 = unigram, 2 = bigram, 3 = trigram. Counts
    for every lower order are kept as well, because all four smoothing methods
    need them for backoff or interpolation.
    """

    def __init__(
        self,
        order: int | None = None,
        smoothing: str | None = None,
        laplace_k: float | None = None,
        backoff_alpha: float | None = None,
        discount: float | None = None,
        bos: str | None = None,
        eos: str | None = None,
        unk: str | None = None,
    ) -> None:
        self.order = order if order is not None else get("ngram.max_order", 3)
        self.smoothing = smoothing if smoothing is not None else get(
            "ngram.smoothing", "stupid_backoff"
        )
        if self.order < 1:
            raise ValueError(f"order must be >= 1, got {self.order}")
        if self.smoothing not in SMOOTHING_METHODS:
            raise ValueError(
                f"unknown smoothing {self.smoothing!r}; "
                f"expected one of {SMOOTHING_METHODS}"
            )

        self.laplace_k = laplace_k if laplace_k is not None else get("ngram.laplace_k", 1.0)
        self.backoff_alpha = backoff_alpha if backoff_alpha is not None else get(
            "ngram.backoff_alpha", 0.4
        )
        self.discount = discount if discount is not None else get(
            "ngram.kneser_ney_discount", 0.75
        )

        self.bos = bos if bos is not None else get("preprocessing.bos_token", "<s>")
        self.eos = eos if eos is not None else get("preprocessing.eos_token", "</s>")
        self.unk = unk if unk is not None else get("preprocessing.unk_token", "<UNK>")

        # counts[k][context] -> Counter(word -> count), where len(context) == k-1
        self.counts: dict[int, dict[Context, Counter]] = {
            k: defaultdict(Counter) for k in range(1, self.order + 1)
        }
        # Continuation counts for Kneser-Ney, filled in by _index().
        self.continuations: dict[int, dict[Context, Counter]] = {}
        # Cached context totals. Recomputing sum(counter.values()) per query
        # is O(continuations per context) and a frequent context such as
        # ("the",) has thousands -- far too slow for the app's ~50 ms budget.
        self.totals: dict[int, dict[Context, int]] = {}
        self.cont_totals: dict[int, dict[Context, int]] = {}
        self.vocab: set[str] = set()
        self.total_tokens = 0
        self.pruned = False

    # ------------------------------------------------------------------ fit

    def fit(self, sentences: Iterable[Sentence]) -> "NgramModel":
        """Count every k-gram for k = 1..order.

        Each sentence is padded once with ``order-1`` BOS markers and one EOS,
        and every order is extracted from that same padded sequence. Padding
        per-order instead would give the lower orders a different number of
        boundary tokens, and the backoff levels would then disagree about how
        often a sentence starts.
        """
        for sentence in sentences:
            padded = pad_sentence(sentence, self.order, bos=self.bos, eos=self.eos)
            self.vocab.update(padded)
            self.total_tokens += len(padded) - (self.order - 1)

            for k in range(1, self.order + 1):
                counts_k = self.counts[k]
                # Start at order-1 so every order sees the same predicted
                # positions; earlier starts would count BOS-only k-grams that
                # the higher orders never see.
                start = self.order - 1
                for i in range(start, len(padded)):
                    context = tuple(padded[i - k + 1:i])
                    counts_k[context][padded[i]] += 1

        self._index()
        return self

    def _index(self) -> None:
        """Continuation counts N1+(• w) — how many distinct words precede a gram.

        This is what makes Kneser-Ney work. "Francisco" is frequent, but it
        follows only "San", so its *continuation* count is 1 and KN refuses to
        predict it after arbitrary contexts. Raw frequency cannot express that.
        """
        self.continuations = {k: defaultdict(Counter) for k in range(1, self.order)}
        for k in range(2, self.order + 1):
            target = self.continuations[k - 1]
            for context, counter in self.counts[k].items():
                # Drop the leftmost token: the remaining gram is the one whose
                # distinct predecessors we are counting.
                suffix_context = context[1:]
                for word in counter:
                    target[suffix_context][word] += 1

        self.totals = {
            k: {ctx: sum(counter.values()) for ctx, counter in table.items()}
            for k, table in self.counts.items()
        }
        self.cont_totals = {
            k: {ctx: sum(counter.values()) for ctx, counter in table.items()}
            for k, table in self.continuations.items()
        }

    # ------------------------------------------------------------- scoring

    def _context_for(self, context: Sequence[str], k: int) -> Context:
        """Trim (or BOS-pad) a context to the k-1 tokens order k needs."""
        needed = k - 1
        if needed == 0:
            return ()
        trimmed = tuple(context[-needed:])
        if len(trimmed) < needed:
            trimmed = (self.bos,) * (needed - len(trimmed)) + trimmed
        return trimmed

    def prob(self, word: str, context: Sequence[str] = ()) -> float:
        """Probability (or, for stupid backoff, score) of ``word`` after ``context``."""
        word = word if word in self.vocab else self.unk
        if self.smoothing == "mle":
            return self._mle(word, context)
        if self.smoothing == "laplace":
            return self._laplace(word, context)
        if self.smoothing == "stupid_backoff":
            return self._stupid_backoff(word, context)
        return self._kneser_ney(word, context)

    def logprob(self, word: str, context: Sequence[str] = ()) -> float:
        p = self.prob(word, context)
        return math.log(p) if p > 0 else -math.inf

    def _mle(self, word: str, context: Sequence[str]) -> float:
        ctx = self._context_for(context, self.order)
        counter = self.counts[self.order].get(ctx)
        if not counter:
            return 0.0
        total = self.totals[self.order][ctx]
        return counter[word] / total if total else 0.0

    def _laplace(self, word: str, context: Sequence[str]) -> float:
        ctx = self._context_for(context, self.order)
        counter = self.counts[self.order].get(ctx) or Counter()
        total = self.totals[self.order].get(ctx, 0)
        k = self.laplace_k
        v = len(self.vocab)
        return (counter[word] + k) / (total + k * v) if v else 0.0

    def _stupid_backoff(self, word: str, context: Sequence[str]) -> float:
        """Brants et al. (2007). Not normalised — a score, not a probability.

        Cheap on purpose: one dict lookup per level, no renormalisation, which
        is what keeps candidate generation inside the app's keystroke budget.
        """
        for k in range(self.order, 1, -1):
            ctx = self._context_for(context, k)
            counter = self.counts[k].get(ctx)
            if counter and counter[word] > 0:
                total = self.totals[k][ctx]
                weight = self.backoff_alpha ** (self.order - k)
                return weight * counter[word] / total
        unigrams = self.counts[1].get((), Counter())
        total = self.totals[1].get((), 0)
        if not total:
            return 0.0
        weight = self.backoff_alpha ** (self.order - 1)
        # An unseen unigram still needs a floor, or the score hits zero and the
        # candidate drops out of the ranking entirely.
        count = unigrams.get(word, 0)
        if count == 0:
            return weight * 1.0 / (total + len(self.vocab))
        return weight * count / total

    def _kneser_ney(self, word: str, context: Sequence[str]) -> float:
        return self._kn_recurse(word, context, self.order, highest=True)

    def _kn_recurse(
        self, word: str, context: Sequence[str], k: int, highest: bool
    ) -> float:
        d = self.discount
        v = len(self.vocab) or 1

        if k == 1:
            if highest:
                counter = self.counts[1].get((), Counter())
                total = self.totals[1].get((), 0)
            else:
                counter = self.continuations.get(1, {}).get((), Counter())
                total = self.cont_totals.get(1, {}).get((), 0)
            if not total:
                return 1.0 / v
            n1plus = len(counter)
            lam = d * n1plus / total
            # Interpolate with the uniform distribution so nothing is ever zero.
            return max(counter[word] - d, 0.0) / total + lam / v

        ctx = self._context_for(context, k)
        if highest:
            counter = self.counts[k].get(ctx)
            total = self.totals[k].get(ctx, 0)
        else:
            counter = self.continuations.get(k, {}).get(ctx)
            total = self.cont_totals.get(k, {}).get(ctx, 0)
        if not counter or not total:
            # No evidence at this order: fall straight through to the next.
            return self._kn_recurse(word, context, k - 1, highest=False)

        n1plus = len(counter)
        lam = d * n1plus / total
        lower = self._kn_recurse(word, context, k - 1, highest=False)
        return max(counter[word] - d, 0.0) / total + lam * lower

    @property
    def is_normalized(self) -> bool:
        """Whether :meth:`prob` sums to 1 over the vocabulary."""
        return self.smoothing in NORMALIZED_METHODS

    # ---------------------------------------------------------- evaluation

    def perplexity(self, sentences: Iterable[Sentence]) -> float:
        """Perplexity over held-out sentences.

        Returns ``inf`` when any token gets probability zero — which is exactly
        what MLE does on unseen n-grams, and the reason the other three
        smoothing methods exist.

        BOS markers are not scored (nothing predicts them); EOS is, because
        knowing where a sentence ends is part of what the model has learned.
        """
        log_sum = 0.0
        n = 0
        for sentence in sentences:
            padded = pad_sentence(sentence, self.order, bos=self.bos, eos=self.eos)
            for i in range(self.order - 1, len(padded)):
                lp = self.logprob(padded[i], padded[:i])
                if lp == -math.inf:
                    return math.inf
                log_sum += lp
                n += 1
        if n == 0:
            return math.inf
        return math.exp(-log_sum / n)

    def distribution_mass(self, context: Sequence[str] = ()) -> float:
        """Total score assigned across the whole vocabulary — 1.0 iff normalised.

        Used in the tests and the notebook to *show* that stupid backoff is not
        a distribution, rather than merely asserting it.
        """
        return sum(self.prob(w, context) for w in self.vocab)

    # ---------------------------------------------------------- prediction

    def candidates(
        self,
        context: Sequence[str] = (),
        k: int = 10,
        exclude: Iterable[str] = (),
    ) -> list[tuple[str, float]]:
        """Top-k next words, highest score first.

        Only words actually observed after some suffix of the context are
        considered. Scoring the entire vocabulary would be correct and far too
        slow — the whole point of the backoff structure is that the observed
        continuations are a short list.

        ``<UNK>`` is excluded along with the boundary markers. It is a frequent
        *class* — on Brown it outranks every real word after ``in the`` — so
        leaving it in means the app's top suggestion is a literal ``<UNK>``.
        The rare words it stands for are exactly the ones the Phase 3 user
        model is there to learn.
        """
        blocked = set(exclude) | {self.bos, self.unk}
        seen: dict[str, float] = {}
        for order in range(self.order, 0, -1):
            ctx = self._context_for(context, order)
            counter = self.counts[order].get(ctx)
            if not counter:
                continue
            for word in counter:
                if word in blocked or word in seen:
                    continue
                seen[word] = self.prob(word, context)
            if len(seen) >= k * 5:
                break
        ranked = sorted(seen.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:k]

    def generate(
        self,
        max_length: int = 25,
        context: Sequence[str] | None = None,
        seed: int | None = None,
    ) -> Sentence:
        """Sample a sentence.

        Sampling is over the observed continuations of the context rather than
        the whole smoothed vocabulary: smoothing spreads a little mass over
        every word, and sampling from that produces word salad even when the
        model is good.
        """
        rng = random.Random(seed)
        history = list(context) if context else [self.bos] * (self.order - 1)
        out: Sentence = []

        for _ in range(max_length):
            counter = None
            for order in range(self.order, 0, -1):
                ctx = self._context_for(history, order)
                candidate = self.counts[order].get(ctx)
                if candidate:
                    counter = candidate
                    break
            if not counter:
                break

            words = [w for w in counter if w != self.bos]
            if not words:
                break
            weights = [counter[w] for w in words]
            word = rng.choices(words, weights=weights, k=1)[0]
            if word == self.eos:
                break
            out.append(word)
            history.append(word)
        return out

    # ------------------------------------------------------------- pruning

    def prune(
        self,
        min_count: int | None = None,
        max_continuations: int | None = None,
    ) -> "NgramModel":
        """Drop rare n-grams and cap continuations per context, in place.

        Unigrams are never pruned — they are the vocabulary and the final
        backoff level, so removing them would put holes in the distribution
        rather than shrink it.

        Pruning changes the distribution: the surviving counts are renormalised
        against a smaller total, so probabilities shift. That cost is real and
        gets measured (size vs. perplexity) rather than assumed away.
        """
        if min_count is None:
            min_count = get("ngram.pruning.min_count", 2)
        if max_continuations is None:
            max_continuations = get("ngram.pruning.max_continuations", 50)

        for k in range(2, self.order + 1):
            kept: dict[Context, Counter] = defaultdict(Counter)
            for context, counter in self.counts[k].items():
                survivors = Counter(
                    {w: c for w, c in counter.items() if c >= min_count}
                )
                if max_continuations and len(survivors) > max_continuations:
                    survivors = Counter(dict(survivors.most_common(max_continuations)))
                if survivors:
                    kept[context] = survivors
            self.counts[k] = kept

        self._index()
        self.pruned = True
        return self

    # ------------------------------------------------------------ reporting

    def stats(self) -> ModelStats:
        per_order = {
            k: sum(len(counter) for counter in self.counts[k].values())
            for k in range(1, self.order + 1)
        }
        return ModelStats(
            order=self.order,
            smoothing=self.smoothing,
            vocab_size=len(self.vocab),
            ngram_counts=per_order,
            total_ngrams=sum(per_order.values()),
        )

    def __repr__(self) -> str:
        return (
            f"NgramModel(order={self.order}, smoothing={self.smoothing!r}, "
            f"vocab={len(self.vocab)}, ngrams={self.stats().total_ngrams})"
        )

    # ------------------------------------------------------- serialisation

    def to_dict(self) -> dict:
        return {
            "order": self.order,
            "smoothing": self.smoothing,
            "laplace_k": self.laplace_k,
            "backoff_alpha": self.backoff_alpha,
            "discount": self.discount,
            "bos": self.bos,
            "eos": self.eos,
            "unk": self.unk,
            "vocab": self.vocab,
            "total_tokens": self.total_tokens,
            "pruned": self.pruned,
            "counts": {
                k: {ctx: dict(counter) for ctx, counter in table.items()}
                for k, table in self.counts.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NgramModel":
        model = cls(
            order=data["order"],
            smoothing=data["smoothing"],
            laplace_k=data["laplace_k"],
            backoff_alpha=data["backoff_alpha"],
            discount=data["discount"],
            bos=data["bos"],
            eos=data["eos"],
            unk=data["unk"],
        )
        model.vocab = set(data["vocab"])
        model.total_tokens = data["total_tokens"]
        model.pruned = data.get("pruned", False)
        model.counts = {
            int(k): defaultdict(Counter, {ctx: Counter(c) for ctx, c in table.items()})
            for k, table in data["counts"].items()
        }
        model._index()
        return model

    def save(self, path: str | Path) -> Path:
        target = resolve_path(path, create=True)
        with target.open("wb") as fh:
            pickle.dump(self.to_dict(), fh, protocol=pickle.HIGHEST_PROTOCOL)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "NgramModel":
        source = resolve_path(path)
        with source.open("rb") as fh:
            return cls.from_dict(pickle.load(fh))


# --------------------------------------------------------------------------
# Experiment helpers (Phase 2 deliverables)
# --------------------------------------------------------------------------

def train_model(
    sentences: Iterable[Sentence],
    order: int,
    smoothing: str,
    **kwargs,
) -> NgramModel:
    return NgramModel(order=order, smoothing=smoothing, **kwargs).fit(sentences)


def perplexity_sweep(
    train: Sequence[Sentence],
    held_out: Sequence[Sentence],
    orders: Sequence[int] = (1, 2, 3),
    methods: Sequence[str] = SMOOTHING_METHODS,
) -> list[dict]:
    """Perplexity for every (order, smoothing) pair — the Phase 2 headline table.

    Counting is done once per order and the smoothing method swapped on the
    fitted model, because refitting per method would multiply the runtime by
    four for identical counts.
    """
    rows = []
    for order in orders:
        base = NgramModel(order=order, smoothing="mle").fit(train)
        for method in methods:
            base.smoothing = method
            rows.append({
                "order": order,
                "smoothing": method,
                "perplexity": base.perplexity(held_out),
                "normalized": base.is_normalized,
                "ngrams": base.stats().total_ngrams,
            })
    return rows


def pruning_report(
    train: Sequence[Sentence],
    held_out: Sequence[Sentence],
    order: int = 3,
    smoothing: str = "stupid_backoff",
    settings: Sequence[tuple[int, int]] = ((1, 0), (2, 50), (3, 20), (5, 10)),
    tmp_path: str | Path | None = None,
) -> list[dict]:
    """Size vs. perplexity for a range of pruning settings.

    ``(min_count, max_continuations)``; ``(1, 0)`` is the unpruned baseline.
    """
    rows = []
    for min_count, max_cont in settings:
        model = NgramModel(order=order, smoothing=smoothing).fit(train)
        if min_count > 1 or max_cont:
            model.prune(min_count=min_count, max_continuations=max_cont)
        payload = pickle.dumps(model.to_dict(), protocol=pickle.HIGHEST_PROTOCOL)
        rows.append({
            "min_count": min_count,
            "max_continuations": max_cont,
            "ngrams": model.stats().total_ngrams,
            "size_mb": len(payload) / 1e6,
            "perplexity": model.perplexity(held_out),
        })
    return rows
