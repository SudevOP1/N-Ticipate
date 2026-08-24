"""Phases 3 and 6 — the prediction engine, and the POS reranker on top of it.

One entry point, :meth:`Predictor.predict`, serving both of the app's modes:

* **word completion** — a partial word is on screen (``rec`` -> ``recommend``),
  candidates come from the prefix trie;
* **next-word prediction** — the user just typed a space, candidates come from
  the n-gram model's continuations.

Both are then scored by the same blended function, so there is one ranking
policy in the system rather than two that drift apart::

    score(w) = lambda * P_user(w | context) + (1 - lambda) * P_base(w | context)

``lambda`` grows with the size of the user profile (see
:attr:`~nticipate.userprofile.UserProfile.weight`), so a fresh install behaves
exactly like the base model and a well-used one leans on what the user actually
writes.

Note honestly that the base model ships with stupid backoff, whose scores are
not normalised, so this interpolation mixes two unnormalised scores. It is a
ranking heuristic, not a probabilistic mixture — which is fine, because ranking
is all the app does with it, but it is not something to write ``P(w|h)`` about
in the report without the caveat.

**Phase 6** adds the POS term. When a fitted :class:`~nticipate.hmm.HMMTagger`
is attached, the blend above is taken into log space and a grammatical prior
is added to it::

    score(w) = log[ lambda * P_user(w | context) + (1 - lambda) * P_base(w | context) ]
               + alpha * log P(tag(w) | tag of the preceding word)

The context is tagged by the HMM's own Viterbi; the candidate contributes a
*context-free* tag guess (:meth:`Predictor.typical_tag`). The asymmetry is
deliberate — the candidate has no right context yet, and running Viterbi once
per candidate would cost fifty decodes per keystroke.

``alpha = 0`` reproduces the Phase 3 ranking exactly: the logarithm is
monotonic, so it reorders nothing.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

from nticipate.config import get, resolve_path
from nticipate.hmm import HMMTagger
from nticipate.ngram import NgramModel
from nticipate.preprocess import Sentence, tokenize, truecase_word
from nticipate.trie import Trie
from nticipate.userprofile import UserProfile


#: A token that is entirely punctuation -- no letters or digits anywhere.
PUNCTUATION_RE = re.compile(r"^[^\w]+$", re.UNICODE)

#: Ceiling on the per-predictor tag caches. Both are keyed by things the user
#: types, so they are unbounded in principle; in practice a few thousand
#: entries cover a session and the cap only matters for the eval loops, which
#: sweep whole corpora through one predictor.
CACHE_LIMIT = 50_000


class Mode(str, Enum):
    COMPLETION = "completion"
    NEXT_WORD = "next_word"


@dataclass(frozen=True)
class Suggestion:
    """One ranked suggestion, ready to display."""

    word: str          # truecased surface form, what the overlay shows
    score: float
    source: str        # "base" | "user"
    mode: Mode
    #: The tag the Phase 6 reranker assumed for this word; ``None`` when
    #: reranking is off. Carried so the notebook can explain a ranking and the
    #: Phase 7 overlay can colour by part of speech.
    tag: str | None = None

    def __str__(self) -> str:
        return self.word


class Predictor:
    """Blends the base n-gram model, the prefix trie and the user profile."""

    def __init__(
        self,
        model: NgramModel,
        trie: Trie | None = None,
        truecase: dict[str, str] | None = None,
        profile: UserProfile | None = None,
        max_suggestions: int | None = None,
        min_prefix_len: int | None = None,
        candidate_pool: int | None = None,
        personalization: bool | None = None,
        suggest_punctuation: bool | None = None,
        tagger: HMMTagger | None = None,
        rerank: bool | None = None,
        rerank_alpha: float | None = None,
        tag_context_size: int | None = None,
        unknown_tag_penalty: float | None = None,
        score_floor: float | None = None,
    ) -> None:
        self.model = model
        self.truecase = truecase or {}
        self.trie = trie if trie is not None else self._trie_from_model(model)

        self.max_suggestions = max_suggestions if max_suggestions is not None else get(
            "prediction.max_suggestions", 3
        )
        self.min_prefix_len = min_prefix_len if min_prefix_len is not None else get(
            "prediction.min_prefix_len", 1
        )
        self.candidate_pool = candidate_pool if candidate_pool is not None else get(
            "prediction.candidate_pool", 50
        )
        self.suggest_punctuation = (
            suggest_punctuation
            if suggest_punctuation is not None
            else get("prediction.suggest_punctuation", True)
        )
        enabled = personalization if personalization is not None else get(
            "prediction.personalization.enabled", True
        )
        self.profile = profile if profile is not None else (UserProfile() if enabled else None)

        # ---- Phase 6 -----------------------------------------------------
        self.rerank = rerank if rerank is not None else get("reranking.enabled", True)
        self.rerank_alpha = rerank_alpha if rerank_alpha is not None else get(
            "reranking.alpha", 0.3
        )
        self.tag_context_size = tag_context_size if tag_context_size is not None else get(
            "reranking.tag_context_size", 2
        )
        self.unknown_tag_penalty = (
            unknown_tag_penalty if unknown_tag_penalty is not None
            else get("reranking.unknown_tag_penalty", -2.0)
        )
        self.score_floor = score_floor if score_floor is not None else get(
            "prediction.score_floor", -30.0
        )
        self.tagger: HMMTagger | None = None
        self._tag_index: dict[str, int] = {}
        self._log_tag_prior: np.ndarray = np.zeros(0)
        self._context_tag_cache: dict[tuple[str, ...], str] = {}
        self._word_tag_cache: dict[str, str] = {}
        if tagger is not None:
            self.attach_tagger(tagger)

    @staticmethod
    def _trie_from_model(model: NgramModel) -> Trie:
        """Build the completion trie from the model's unigram counts.

        Boundary markers and ``<UNK>`` are left out: they are not words anyone
        can type a prefix of, and ``<UNK>`` in particular is a frequent class
        that would otherwise dominate every completion list.
        """
        skip = {model.bos, model.eos, model.unk}
        counts = {
            word: count
            for word, count in model.counts[1][()].items()
            if word not in skip and count > 0
        }
        return Trie.from_counts(counts)

    # ------------------------------------------------------------ loading

    @classmethod
    def from_paths(
        cls,
        model_path: str | Path,
        corpus_path: str | Path | None = None,
        profile_path: str | Path | None = None,
        tagger_path: str | Path | None = None,
        truecase_path: str | Path | None = None,
        **kwargs,
    ) -> "Predictor":
        """Load a shipped model, its truecase map, the profile and the tagger.

        ``truecase_path`` is the cheap way in: a map written by
        :meth:`Corpus.save_truecase`, a few MB. ``corpus_path`` is the fallback
        for callers that only have the full corpus, and parses the splits and
        the vocab the predictor then throws away.
        """
        model = NgramModel.load(model_path)
        truecase: dict[str, str] = {}
        if truecase_path is not None:
            from nticipate.preprocess import load_truecase_map

            truecase = load_truecase_map(truecase_path)
        elif corpus_path is not None:
            from nticipate.preprocess import Corpus

            truecase = Corpus.load(corpus_path).truecase
        profile = UserProfile.load(profile_path) if profile_path is not None else None
        tagger = HMMTagger.load(tagger_path) if tagger_path is not None else None
        return cls(model, truecase=truecase, profile=profile, tagger=tagger, **kwargs)

    # ----------------------------------------------------------- reranking

    def attach_tagger(self, tagger: HMMTagger | None) -> "Predictor":
        """Give the predictor a fitted tagger (or take it away again).

        The tag index and the tag prior are derived once here rather than once
        per candidate, and the caches are dropped because they hold the old
        tagger's answers.
        """
        self._context_tag_cache = {}
        self._word_tag_cache = {}
        self.tagger = tagger
        if tagger is None or not tagger.is_fitted:
            self._tag_index = {}
            self._log_tag_prior = np.zeros(0)
            return self
        self._tag_index = {tag: i for i, tag in enumerate(tagger.tags)}
        counts = np.array([tagger.tag_counts[t] for t in tagger.tags], dtype=np.float64)
        self._log_tag_prior = np.log(counts) - math.log(counts.sum())
        return self

    @property
    def rerank_active(self) -> bool:
        """Whether :meth:`predict` will add the POS term."""
        return bool(self.rerank and self.tagger is not None and self.tagger.is_fitted)

    def context_tag(self, context: Sequence[str]) -> str | None:
        """The tag of the word immediately before the cursor.

        The HMM is a *bigram* model over tags, so exactly one preceding tag
        conditions the reranking term. ``reranking.tag_context_size`` is not
        that number -- it is how many preceding tokens get decoded to obtain
        it. Viterbi's choice for the last token depends on its left context, so
        decoding a two-token window is not the same as decoding one token; it
        is, however, nearly the same as decoding the whole buffer, at a
        fraction of the cost. Notebook 06 measures how often the window size
        changes the answer.

        This deliberately does **not** call :meth:`~nticipate.hmm.HMMTagger.viterbi`,
        which decodes a finished *sentence* and therefore adds the transition
        into the end-of-sentence state. A buffer the user is still typing has
        not ended, and charging it that transition is not a small effect: the
        end state is reached overwhelmingly from punctuation, so ``i would``
        came back tagged ``PRON .`` -- the tagger preferring to believe
        ``would`` was a full stop over believing the sentence continued. Taking
        the arg-max of the last trellis column instead drops that term and
        returns ``VERB``.

        ``None`` when there is no context at all, in which case the candidate
        is scored against the tagger's initial distribution instead -- which is
        exactly right at the start of a sentence.
        """
        window = tuple(context[-self.tag_context_size:]) if self.tag_context_size else ()
        if not window:
            return None
        cached = self._context_tag_cache.get(window)
        if cached is None:
            # column(-1) is sorted best-first, so the first key is the arg-max
            # of the last column -- the same recursion Viterbi runs, minus the
            # end-of-sentence transition it would apply afterwards.
            cached = next(iter(self.tagger.trellis(list(window)).column(-1)))
            if len(self._context_tag_cache) < CACHE_LIMIT:
                self._context_tag_cache[window] = cached
        return cached

    def typical_tag(self, word: str) -> str:
        """The candidate's context-free tag guess: ``argmax_t P(t | word)``.

        By Bayes, dropping the constant ``P(word)``, that is
        ``argmax_t P(word | t) P(t)`` -- the tagger's own emission column plus
        the tag prior. Words the tagger never saw fall through
        :meth:`~nticipate.hmm.HMMTagger.emission_column` to the unseen-word
        machinery, so the suffix heuristics of Phases 4 and 5 apply here for
        free, in both scripts.

        Context-free on purpose. The candidate sits at the end of the buffer
        with nothing to its right, and tagging it in context would mean one
        Viterbi decode per candidate -- fifty per keystroke, well outside the
        debounce budget, for a term that only breaks ties.
        """
        cached = self._word_tag_cache.get(word)
        if cached is not None:
            return cached
        posterior = self.tagger.emission_column(word) + self._log_tag_prior
        tag = self.tagger.tags[int(posterior.argmax())]
        if len(self._word_tag_cache) < CACHE_LIMIT:
            self._word_tag_cache[word] = tag
        return tag

    def tag_score(self, word: str, tag: str, previous_tag: str | None) -> float:
        """``log P(tag | previous_tag)``, penalised when the tag was guessed.

        A word the tagger never saw got its tag from spelling alone, which is
        wrong often enough that its POS term should not carry full weight --
        hence ``reranking.unknown_tag_penalty``, a flat handicap rather than a
        veto.
        """
        index = self._tag_index[tag]
        if previous_tag is None:
            score = float(self.tagger.log_initial[index])
        else:
            score = float(self.tagger.log_transition[self._tag_index[previous_tag], index])
        if not self.tagger.knows(word):
            score += self.unknown_tag_penalty
        return score

    def _log_score(self, score: float) -> float:
        """Blended probability -> log, with a floor instead of ``-inf``.

        A blend of exactly zero is common and harmless: a fresh user profile
        has weight zero, so a word only the user's own model knows scores zero.
        ``prediction.score_floor`` parks those candidates at the bottom of the
        ranking without poisoning the arithmetic above them.
        """
        return math.log(score) if score > 0.0 else self.score_floor

    # ---------------------------------------------------------- prediction

    @property
    def context_size(self) -> int:
        """Tokens of left context the model can use: ``order - 1``."""
        return max(0, self.model.order - 1)

    def split_buffer(self, text: str) -> tuple[Sentence, str]:
        """Split a typed buffer into (context tokens, current prefix).

        A trailing space means the current word is finished, so the mode is
        next-word prediction; otherwise the final token is the prefix being
        typed. This is the function the Phase 7 keystroke hook calls.
        """
        if not text:
            return [], ""
        tokens = tokenize(text)
        if text[-1].isspace():
            prefix = ""
        else:
            prefix = tokens[-1] if tokens else ""
            tokens = tokens[:-1]
        return tokens[-self.context_size:] if self.context_size else [], prefix

    def suggest(self, text: str, k: int | None = None) -> list[Suggestion]:
        """Suggest from a raw typed buffer."""
        context, prefix = self.split_buffer(text)
        return self.predict(context, prefix, k)

    def predict(
        self,
        context: Sequence[str] = (),
        prefix: str = "",
        k: int | None = None,
    ) -> list[Suggestion]:
        """Rank suggestions for the given context and partial word."""
        k = k or self.max_suggestions
        context = [t.lower() for t in context][-self.context_size:] if self.context_size else []
        prefix_lower = prefix.lower()

        if prefix_lower:
            if len(prefix_lower) < self.min_prefix_len:
                return []
            mode = Mode.COMPLETION
            words = self._completion_candidates(prefix_lower)
        else:
            mode = Mode.NEXT_WORD
            words = self._next_word_candidates(context)

        if not self.suggest_punctuation:
            words = [w for w in words if not PUNCTUATION_RE.match(w)]

        rerank = self.rerank_active
        previous_tag = self.context_tag(context) if rerank else None

        lam = self.profile.weight if self.profile else 0.0
        scored: list[Suggestion] = []
        for word in words:
            base = self.model.prob(word, context)
            user = self.profile.prob(word, context) if self.profile else 0.0
            score = lam * user + (1.0 - lam) * base
            tag = None
            if rerank:
                tag = self.typical_tag(word)
                score = self._log_score(score) + self.rerank_alpha * self.tag_score(
                    word, tag, previous_tag
                )
            scored.append(
                Suggestion(
                    word=truecase_word(word, self.truecase),
                    score=score,
                    source="user" if user > 0 else "base",
                    mode=mode,
                    tag=tag,
                )
            )

        scored.sort(key=lambda s: (-s.score, s.word))
        return scored[:k]

    def _completion_candidates(self, prefix: str) -> list[str]:
        """Words starting with ``prefix``, from the base trie and the user's.

        The user's trie is consulted separately rather than merged into the
        base one, because the whole point of the personalisation layer is the
        vocabulary the base model threw away: names, jargon, identifiers. Those
        words are not in the base trie at all, so a merged lookup would never
        surface them.
        """
        pool = self.candidate_pool
        words = {w for w, _ in self.trie.complete(prefix, k=pool)}
        if self.profile:
            words |= {w for w, _ in self.profile.complete(prefix, k=pool)}
        return list(words)

    def _next_word_candidates(self, context: Sequence[str]) -> list[str]:
        pool = self.candidate_pool
        words = {w for w, _ in self.model.candidates(context, k=pool)}
        if self.profile:
            words |= {w for w, _ in self.profile.candidates(context, k=pool)}
        skip = {self.model.bos, self.model.eos, self.model.unk}
        return [w for w in words if w not in skip]

    # ------------------------------------------------------------ learning

    def learn(self, text: str) -> int:
        """Feed accepted or typed text back into the user profile."""
        if not self.profile:
            return 0
        return self.profile.observe_text(text)

    def __repr__(self) -> str:
        lam = self.profile.weight if self.profile else 0.0
        rerank = f"alpha={self.rerank_alpha}" if self.rerank_active else "off"
        return (
            f"Predictor(order={self.model.order}, "
            f"smoothing={self.model.smoothing!r}, trie={len(self.trie)}, "
            f"lambda={lam:.3f}, rerank={rerank})"
        )


# --------------------------------------------------------------------------
# Evaluation (Phase 3 deliverables)
# --------------------------------------------------------------------------

def _is_scorable(token: str, model: NgramModel) -> bool:
    """Whether a target token is a fair thing to ask the predictor for.

    ``<UNK>`` targets are excluded. The predictor deliberately never suggests
    ``<UNK>``, so counting those positions would report a miss for a word the
    engine is designed not to offer, and understate accuracy for reasons that
    have nothing to do with the model's quality.
    """
    return token not in {model.bos, model.eos, model.unk}


def hit_at_k(
    predictor: Predictor,
    sentences: Sequence[Sentence],
    ks: Sequence[int] = (1, 3, 5),
    limit: int | None = None,
) -> dict:
    """Next-word accuracy: how often the true next word is in the top k."""
    max_k = max(ks)
    hits = {k: 0 for k in ks}
    total = 0

    for sentence in sentences[:limit]:
        for i, target in enumerate(sentence):
            if not _is_scorable(target, predictor.model):
                continue
            context = sentence[max(0, i - predictor.context_size):i]
            ranked = [s.word.lower() for s in predictor.predict(context, "", k=max_k)]
            total += 1
            for k in ks:
                if target in ranked[:k]:
                    hits[k] += 1

    return {
        "positions": total,
        **{f"hit@{k}": (hits[k] / total if total else 0.0) for k in ks},
    }


def completion_hit_at_k(
    predictor: Predictor,
    sentences: Sequence[Sentence],
    ks: Sequence[int] = (1, 3, 5),
    prefix_len: int = 2,
    limit: int | None = None,
) -> dict:
    """Completion accuracy after ``prefix_len`` characters have been typed."""
    max_k = max(ks)
    hits = {k: 0 for k in ks}
    total = 0

    for sentence in sentences[:limit]:
        for i, target in enumerate(sentence):
            if not _is_scorable(target, predictor.model) or len(target) <= prefix_len:
                continue
            context = sentence[max(0, i - predictor.context_size):i]
            ranked = [
                s.word.lower()
                for s in predictor.predict(context, target[:prefix_len], k=max_k)
            ]
            total += 1
            for k in ks:
                if target in ranked[:k]:
                    hits[k] += 1

    return {
        "positions": total,
        "prefix_len": prefix_len,
        **{f"hit@{k}": (hits[k] / total if total else 0.0) for k in ks},
    }


def keystroke_savings(
    predictor: Predictor,
    sentences: Sequence[Sentence],
    k: int | None = None,
    limit: int | None = None,
) -> dict:
    """Simulate typing and measure the fraction of keystrokes saved.

    The model: typing a word costs one keystroke per character plus a space.
    Accepting a suggestion costs the characters typed so far plus one accept
    key. A suggestion is only taken when it actually saves a keystroke —
    accepting a three-letter word after typing two letters saves nothing and a
    real user would not do it.

    This is the headline product number: perplexity says the model is good,
    keystroke savings says the app is useful.
    """
    k = k or predictor.max_suggestions
    typed = baseline = 0
    accepted = words = 0

    for sentence in sentences[:limit]:
        for i, target in enumerate(sentence):
            if not _is_scorable(target, predictor.model):
                continue
            words += 1
            cost_full = len(target) + 1  # characters plus the space
            baseline += cost_full
            context = sentence[max(0, i - predictor.context_size):i]

            cost = cost_full
            for p in range(predictor.min_prefix_len, len(target)):
                if p + 1 >= cost_full:
                    break  # accepting here would cost more than typing it out
                ranked = [s.word.lower() for s in predictor.predict(context, target[:p], k=k)]
                if target in ranked:
                    cost = p + 1
                    accepted += 1
                    break
            typed += cost

    return {
        "words": words,
        "accepted": accepted,
        "acceptance_rate": accepted / words if words else 0.0,
        "keystrokes_typed": typed,
        "keystrokes_baseline": baseline,
        "savings": 1 - (typed / baseline) if baseline else 0.0,
    }


def latency_stats(
    predictor: Predictor,
    contexts: Iterable[tuple[Sequence[str], str]],
    repeats: int = 1,
    percentiles: Sequence[int] = (50, 95, 99),
) -> dict:
    """Per-call latency in milliseconds, at the given percentiles."""
    samples: list[float] = []
    contexts = list(contexts)
    for _ in range(repeats):
        for context, prefix in contexts:
            start = time.perf_counter()
            predictor.predict(context, prefix)
            samples.append((time.perf_counter() - start) * 1000)

    if not samples:
        return {"calls": 0}
    samples.sort()
    out = {"calls": len(samples), "mean_ms": sum(samples) / len(samples)}
    for p in percentiles:
        index = min(len(samples) - 1, int(len(samples) * p / 100))
        out[f"p{p}_ms"] = samples[index]
    return out


# --------------------------------------------------------------------------
# The Phase 6 ablation
# --------------------------------------------------------------------------

@contextmanager
def reranking(
    predictor: Predictor,
    enabled: bool,
    alpha: float | None = None,
) -> Iterator[Predictor]:
    """Temporarily force the reranker on or off, then put it back.

    The ablation runs the same predictor twice rather than building two, so
    the trie, the profile and both caches are identical across the arms and
    nothing but the POS term can move the numbers.
    """
    previous = (predictor.rerank, predictor.rerank_alpha)
    predictor.rerank = enabled
    if alpha is not None:
        predictor.rerank_alpha = alpha
    try:
        yield predictor
    finally:
        predictor.rerank, predictor.rerank_alpha = previous


def rerank_ablation(
    predictor: Predictor,
    sentences: Sequence[Sentence],
    ks: Sequence[int] = (1, 3, 5),
    limit: int | None = None,
    prefix_len: int = 2,
    alpha: float | None = None,
) -> dict:
    """hit@k with and without the POS term -- the core Phase 6 deliverable.

    Both prediction modes are measured, because there is every reason to
    expect them to differ: next-word prediction is where a tag prior has room
    to help, while completion has already been handed the first letters, which
    is a far stronger constraint than a part of speech.

    A small or mixed delta is a real result at this corpus size, not a failed
    experiment. Report it as measured.
    """
    if predictor.tagger is None or not predictor.tagger.is_fitted:
        raise ValueError("rerank_ablation needs a predictor with a fitted tagger")

    out: dict = {}
    for label, enabled in (("off", False), ("on", True)):
        with reranking(predictor, enabled, alpha if enabled else None):
            out[label] = {
                "next_word": hit_at_k(predictor, sentences, ks, limit),
                "completion": completion_hit_at_k(
                    predictor, sentences, ks, prefix_len, limit
                ),
            }
    out["alpha"] = alpha if alpha is not None else predictor.rerank_alpha
    out["delta"] = {
        task: {
            f"hit@{k}": out["on"][task][f"hit@{k}"] - out["off"][task][f"hit@{k}"]
            for k in ks
        }
        for task in ("next_word", "completion")
    }
    return out


def alpha_sweep(
    predictor: Predictor,
    sentences: Sequence[Sentence],
    alphas: Sequence[float] = (0.0, 0.1, 0.2, 0.3, 0.5, 1.0),
    ks: Sequence[int] = (1, 3, 5),
    limit: int | None = None,
    mode: Mode = Mode.NEXT_WORD,
    prefix_len: int = 2,
) -> list[dict]:
    """hit@k as a function of the POS weight.

    ``alpha = 0`` is the Phase 3 ranking exactly -- the logarithm the reranker
    applies is monotonic, so with a zero weight on the tag term nothing moves.
    That row is the sanity check that the two arms of the ablation differ only
    in the thing being ablated.
    """
    if predictor.tagger is None or not predictor.tagger.is_fitted:
        raise ValueError("alpha_sweep needs a predictor with a fitted tagger")

    rows = []
    for alpha in alphas:
        with reranking(predictor, True, alpha):
            if mode is Mode.NEXT_WORD:
                result = hit_at_k(predictor, sentences, ks, limit)
            else:
                result = completion_hit_at_k(
                    predictor, sentences, ks, prefix_len, limit
                )
        rows.append({"alpha": alpha, "mode": mode.value, **result})
    return rows


def tag_window_disagreement(
    predictor: Predictor,
    sentences: Sequence[Sentence],
    limit: int | None = None,
) -> dict:
    """What the tagging window costs, split from what the task costs.

    :meth:`Predictor.context_tag` decodes only the last few tokens of the
    buffer. Two different things are lost by that, and lumping them together
    would badly overstate the approximation:

    ``vs_prefix``
        The windowed decode against a decode of the *whole buffer so far* —
        same information, more of it, and the end-of-sentence transition
        dropped from both. This is the cost of the window itself, and the only
        part that shrinking ``tag_context_size`` is responsible for.
    ``vs_sentence``
        The windowed decode against a decode of the finished sentence, right
        context and end transition included. Much larger, and not fixable by
        any window: at prediction time the words to the right have not been
        typed yet. It is the price of tagging a live buffer at all, and it
        belongs in the report as a property of the task rather than of this
        code.
    """
    if predictor.tagger is None or not predictor.tagger.is_fitted:
        raise ValueError("tag_window_disagreement needs a predictor with a fitted tagger")

    tagger = predictor.tagger
    positions = vs_prefix = vs_sentence = 0
    for sentence in sentences[:limit]:
        full = tagger.viterbi(sentence)
        for i in range(1, len(sentence)):
            windowed = predictor.context_tag(sentence[:i])
            prefix_tag = next(iter(tagger.trellis(sentence[:i]).column(-1)))
            positions += 1
            vs_prefix += windowed != prefix_tag
            vs_sentence += windowed != full[i - 1]
    return {
        "positions": positions,
        "window": predictor.tag_context_size,
        "vs_prefix": vs_prefix / positions if positions else 0.0,
        "vs_sentence": vs_sentence / positions if positions else 0.0,
    }


def typical_tag_agreement(
    predictor: Predictor,
    sentences: Sequence[Sentence],
    limit: int | None = None,
) -> dict:
    """How often the context-free candidate tag matches the in-context one.

    :meth:`Predictor.typical_tag` guesses a candidate's part of speech from
    the word alone, because tagging every candidate in context would cost one
    Viterbi decode per candidate per keystroke. This is the accuracy of that
    shortcut, measured against the tagger's own in-context decision — the
    ceiling on how much the POS term can be worth, and the first place to look
    when it disappoints.

    The reference here is Viterbi's tag, not a gold tag: the question is what
    the shortcut costs relative to doing the expensive thing, not how good the
    tagger is. Phase 4 already measured that.
    """
    if predictor.tagger is None or not predictor.tagger.is_fitted:
        raise ValueError("typical_tag_agreement needs a predictor with a fitted tagger")

    tagger = predictor.tagger
    tokens = agree = 0
    known_tokens = known_agree = 0
    confusions: Counter = Counter()
    for sentence in sentences[:limit]:
        decoded = tagger.viterbi(sentence)
        for word, in_context in zip(sentence, decoded):
            if not _is_scorable(word, predictor.model):
                continue
            guess = predictor.typical_tag(word)
            tokens += 1
            hit = guess == in_context
            agree += hit
            if tagger.knows(word):
                known_tokens += 1
                known_agree += hit
            if not hit:
                confusions[(in_context, guess)] += 1
    return {
        "tokens": tokens,
        "agreement": agree / tokens if tokens else 0.0,
        "known_agreement": known_agree / known_tokens if known_tokens else 0.0,
        "unknown_tokens": tokens - known_tokens,
        "top_confusions": confusions.most_common(10),
    }
