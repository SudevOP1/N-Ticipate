"""Phase 3 — per-user personalisation layer.

A small n-gram model trained continuously on what the user actually types,
interpolated with the base model at prediction time.

The design decision that matters here is that the user's own vocabulary is
**never** ``<UNK>``-ed. The base model discards everything below
``min_token_freq`` — which is exactly the user's colleagues' names, their
project jargon, their variable names. Those words are rare in Brown and common
in the user's typing, and capturing them is the entire point of this layer.

Storage is a ring buffer over sentences, capped at ``max_user_tokens``. When
the cap is passed, the oldest sentences are evicted and their counts are
subtracted, so the profile tracks how the user writes *now* rather than
accumulating forever.

Privacy: this is the one component that holds the user's own text. It stays on
disk under ``paths.user_profile``, never leaves the machine, and
:meth:`UserProfile.reset` deletes it.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Iterable, Sequence

from nticipate.config import get, resolve_path
from nticipate.preprocess import Sentence, pad_sentence, tokenize_text
from nticipate.trie import Trie

Context = tuple[str, ...]


class UserProfile:
    """An incrementally-trained n-gram model over the user's own text."""

    def __init__(
        self,
        order: int | None = None,
        max_tokens: int | None = None,
        lambda_max: float | None = None,
        lambda_growth_tokens: int | None = None,
        backoff_alpha: float | None = None,
        bos: str | None = None,
        eos: str | None = None,
    ) -> None:
        self.order = order if order is not None else get("ngram.max_order", 3)
        self.max_tokens = max_tokens if max_tokens is not None else get(
            "prediction.personalization.max_user_tokens", 200000
        )
        self.lambda_max = lambda_max if lambda_max is not None else get(
            "prediction.personalization.lambda_max", 0.4
        )
        self.lambda_growth_tokens = (
            lambda_growth_tokens
            if lambda_growth_tokens is not None
            else get("prediction.personalization.lambda_growth_tokens", 5000)
        )
        self.backoff_alpha = backoff_alpha if backoff_alpha is not None else get(
            "ngram.backoff_alpha", 0.4
        )
        self.bos = bos if bos is not None else get("preprocessing.bos_token", "<s>")
        self.eos = eos if eos is not None else get("preprocessing.eos_token", "</s>")

        self.counts: dict[int, dict[Context, Counter]] = {
            k: defaultdict(Counter) for k in range(1, self.order + 1)
        }
        self.totals: dict[int, dict[Context, int]] = {
            k: defaultdict(int) for k in range(1, self.order + 1)
        }
        self.trie = Trie()
        self.sentences: deque[Sentence] = deque()
        self.token_count = 0

    # ----------------------------------------------------------- observing

    def observe_text(self, text: str) -> int:
        """Tokenise raw typed text and learn from it. Returns tokens added."""
        added = 0
        for sentence in tokenize_text(text):
            added += self.observe(sentence)
        return added

    def observe(self, sentence: Sequence[str]) -> int:
        """Learn one sentence. Counting is lowercased, to match the base model."""
        tokens = [t.lower() for t in sentence if t]
        if not tokens:
            return 0

        self.sentences.append(tokens)
        self._apply(tokens, sign=1)
        self.token_count += len(tokens)
        self._evict_if_needed()
        return len(tokens)

    def _apply(self, tokens: Sentence, sign: int) -> None:
        """Add (sign=+1) or subtract (sign=-1) one sentence's counts."""
        padded = pad_sentence(tokens, self.order, bos=self.bos, eos=self.eos)
        for k in range(1, self.order + 1):
            counts_k = self.counts[k]
            totals_k = self.totals[k]
            for i in range(self.order - 1, len(padded)):
                context = tuple(padded[i - k + 1:i])
                word = padded[i]
                counts_k[context][word] += sign
                totals_k[context] += sign
                if counts_k[context][word] <= 0:
                    del counts_k[context][word]
                if totals_k[context] <= 0:
                    counts_k.pop(context, None)
                    totals_k.pop(context, None)

        for token in tokens:
            if sign > 0:
                self.trie.insert(token)
            # Trie entries are deliberately not removed on eviction: a name the
            # user typed once should stay completable even after its n-gram
            # counts have aged out. It costs a few bytes and avoids the app
            # forgetting a colleague's name mid-conversation.

    def _evict_if_needed(self) -> None:
        while self.max_tokens and self.token_count > self.max_tokens and self.sentences:
            oldest = self.sentences.popleft()
            self._apply(oldest, sign=-1)
            self.token_count -= len(oldest)

    # ------------------------------------------------------------- scoring

    def _context_for(self, context: Sequence[str], k: int) -> Context:
        needed = k - 1
        if needed == 0:
            return ()
        trimmed = tuple(t.lower() for t in context[-needed:])
        if len(trimmed) < needed:
            trimmed = (self.bos,) * (needed - len(trimmed)) + trimmed
        return trimmed

    def prob(self, word: str, context: Sequence[str] = ()) -> float:
        """Stupid-backoff score under the user model, or 0.0 if unseen.

        Returning a hard zero for a word the user has never typed is
        deliberate: the interpolation in :class:`~nticipate.predictor.Predictor`
        then falls back entirely to the base model, which is the correct
        behaviour for a profile holding a few thousand tokens.
        """
        word = word.lower()
        for k in range(self.order, 1, -1):
            ctx = self._context_for(context, k)
            counter = self.counts[k].get(ctx)
            if counter and counter.get(word, 0) > 0:
                weight = self.backoff_alpha ** (self.order - k)
                return weight * counter[word] / self.totals[k][ctx]
        unigrams = self.counts[1].get(())
        total = self.totals[1].get((), 0)
        if not unigrams or not total:
            return 0.0
        count = unigrams.get(word, 0)
        if count == 0:
            return 0.0
        return (self.backoff_alpha ** (self.order - 1)) * count / total

    @property
    def weight(self) -> float:
        """Interpolation weight lambda, growing with the evidence available.

        A profile holding fifty tokens should barely influence anything; one
        holding thousands has earned its say. Linear growth to ``lambda_max``
        over ``lambda_growth_tokens``, then flat — the cap exists because the
        user's own text is small and topically narrow, and letting it dominate
        would make the app worse at ordinary English.
        """
        if not self.lambda_growth_tokens:
            return self.lambda_max
        progress = min(1.0, self.token_count / self.lambda_growth_tokens)
        return self.lambda_max * progress

    def candidates(self, context: Sequence[str] = (), k: int = 10) -> list[tuple[str, float]]:
        seen: dict[str, float] = {}
        for order in range(self.order, 0, -1):
            ctx = self._context_for(context, order)
            counter = self.counts[order].get(ctx)
            if not counter:
                continue
            for word in counter:
                if word in (self.bos, self.eos) or word in seen:
                    continue
                seen[word] = self.prob(word, context)
        return sorted(seen.items(), key=lambda kv: (-kv[1], kv[0]))[:k]

    def complete(self, prefix: str, k: int = 10) -> list[tuple[str, int]]:
        return self.trie.complete(prefix.lower(), k=k)

    # ------------------------------------------------------------ lifecycle

    def reset(self) -> None:
        """Forget everything. Backs the tray app's "clear my data" action."""
        self.__init__(
            order=self.order,
            max_tokens=self.max_tokens,
            lambda_max=self.lambda_max,
            lambda_growth_tokens=self.lambda_growth_tokens,
            backoff_alpha=self.backoff_alpha,
            bos=self.bos,
            eos=self.eos,
        )

    def save(self, path: str | Path | None = None) -> Path:
        target = resolve_path(
            path if path is not None else get("paths.user_profile"), create=True
        )
        payload = {
            "order": self.order,
            "max_tokens": self.max_tokens,
            "lambda_max": self.lambda_max,
            "lambda_growth_tokens": self.lambda_growth_tokens,
            "backoff_alpha": self.backoff_alpha,
            "sentences": list(self.sentences),
        }
        with target.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
        return target

    @classmethod
    def load(cls, path: str | Path | None = None) -> "UserProfile":
        """Rebuild from the stored sentences.

        Only the raw sentences are persisted; the counts, totals and trie are
        recomputed on load. That keeps the file small and — more usefully —
        means a change to the counting logic does not silently leave stale
        derived state on disk.
        """
        source = resolve_path(path if path is not None else get("paths.user_profile"))
        if not source.is_file():
            return cls()
        with source.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        profile = cls(
            order=data.get("order"),
            max_tokens=data.get("max_tokens"),
            lambda_max=data.get("lambda_max"),
            lambda_growth_tokens=data.get("lambda_growth_tokens"),
            backoff_alpha=data.get("backoff_alpha"),
        )
        for sentence in data.get("sentences", []):
            profile.observe(sentence)
        return profile

    def stats(self) -> dict:
        return {
            "tokens": self.token_count,
            "sentences": len(self.sentences),
            "vocabulary": len(self.trie),
            "weight": self.weight,
        }

    def __repr__(self) -> str:
        return (
            f"UserProfile(tokens={self.token_count}, "
            f"vocab={len(self.trie)}, lambda={self.weight:.3f})"
        )
