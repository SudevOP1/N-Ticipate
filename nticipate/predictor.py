"""Phase 3 — the prediction engine.

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
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Sequence

from nticipate.config import get, resolve_path
from nticipate.ngram import NgramModel
from nticipate.preprocess import Sentence, tokenize, truecase_word
from nticipate.trie import Trie
from nticipate.userprofile import UserProfile


#: A token that is entirely punctuation -- no letters or digits anywhere.
PUNCTUATION_RE = re.compile(r"^[^\w]+$", re.UNICODE)


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
        **kwargs,
    ) -> "Predictor":
        """Load a shipped model, its truecase map and the user profile."""
        model = NgramModel.load(model_path)
        truecase: dict[str, str] = {}
        if corpus_path is not None:
            from nticipate.preprocess import Corpus

            truecase = Corpus.load(corpus_path).truecase
        profile = UserProfile.load(profile_path) if profile_path is not None else None
        return cls(model, truecase=truecase, profile=profile, **kwargs)

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

        lam = self.profile.weight if self.profile else 0.0
        scored: list[Suggestion] = []
        for word in words:
            base = self.model.prob(word, context)
            user = self.profile.prob(word, context) if self.profile else 0.0
            score = lam * user + (1.0 - lam) * base
            scored.append(
                Suggestion(
                    word=truecase_word(word, self.truecase),
                    score=score,
                    source="user" if user > 0 else "base",
                    mode=mode,
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
        return (
            f"Predictor(order={self.model.order}, "
            f"smoothing={self.model.smoothing!r}, trie={len(self.trie)}, "
            f"lambda={lam:.3f})"
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
