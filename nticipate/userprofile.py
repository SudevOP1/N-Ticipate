"""
Phase 3 (personalization): a small n-gram model trained continuously on
what the user actually types, interpolated with the base corpus model.

lambda (weight on the user model) starts low and grows toward
CFG['predictor']['personalization']['lambda_max'] as more user tokens
are observed -- see config.yaml for the schedule, and current_lambda()
below for the exact curve.

Deliberately does NOT run observed sentences through preprocess.apply_unk
-- the whole point of this model is to capture the user's own
vocabulary (names, jargon, abbreviations) that the base corpus model
would otherwise <UNK> away.
"""

import pickle

from nticipate.config import load_config
from nticipate.ngram import build_ngram_hierarchy

CFG = load_config()

# Token count (in multiples of min_user_tokens) at which lambda saturates
# at lambda_max. Not itself a config.yaml knob -- it's an internal
# curve-shape choice (how fast personalization ramps up), not something
# the app exposes as a user-facing setting.
_LAMBDA_SATURATION_MULTIPLIER = 10


class UserProfile:
    def __init__(self):
        self.sentences: list[list[str]] = []
        self.token_count: int = 0
        self.models: dict[int, "NgramModel"] = {}  # noqa: F821 -- see ngram.NgramModel
        self._dirty: bool = False

    # ------------------------------------------------------------------
    # observing new typed sentences
    # ------------------------------------------------------------------
    def observe_sentence(self, tokens: list[str]) -> None:
        """Update the user model with a sentence the user just finished typing."""
        if not tokens:
            return
        self.sentences.append(list(tokens))
        self.token_count += len(tokens)
        self._dirty = True

    @property
    def vocab(self) -> set[str]:
        """Unique tokens the user has typed, in their original casing --
        used by Predictor for prefix-completion of user-specific words
        (names, jargon) that the trie (built from the base corpus vocab)
        doesn't know about.
        """
        return {tok for sentence in self.sentences for tok in sentence}

    def _ensure_fitted(self) -> None:
        if not self._dirty:
            return
        orders = CFG["ngram"]["orders"]
        # stupid_backoff: cheap, no renormalization, and doesn't need a
        # fixed vocab size the way Laplace does -- important since the
        # user's vocabulary keeps growing as they type.
        self.models = build_ngram_hierarchy(self.sentences, orders=orders, smoothing="stupid_backoff")
        self._dirty = False

    # ------------------------------------------------------------------
    # queries (used by predictor.py's _blend_with_user_model)
    # ------------------------------------------------------------------
    def prob(self, word: str, context: tuple[str, ...], n: int | None = None) -> float:
        self._ensure_fitted()
        if not self.models:
            return 0.0
        n = min(n, max(self.models)) if n is not None else max(self.models)
        if n not in self.models:
            return 0.0
        return self.models[n].prob(word, context)

    def top_k(self, context: tuple[str, ...], k: int = 5, n: int | None = None) -> list[tuple[str, float]]:
        self._ensure_fitted()
        if not self.models:
            return []
        n = min(n, max(self.models)) if n is not None else max(self.models)
        if n not in self.models:
            return []
        return self.models[n].top_k(context, k=k)

    # ------------------------------------------------------------------
    # interpolation weight
    # ------------------------------------------------------------------
    def current_lambda(self) -> float:
        """Interpolation weight based on how much user data we've seen.

        Flat at lambda_start below min_user_tokens (not enough data to
        trust yet), then ramps linearly to lambda_max by
        min_user_tokens * _LAMBDA_SATURATION_MULTIPLIER tokens, then caps.
        """
        cfg = CFG["predictor"]["personalization"]
        if not cfg.get("enabled", True):
            return 0.0

        start = cfg["lambda_start"]
        cap = cfg["lambda_max"]
        min_tokens = cfg["min_user_tokens"]

        if self.token_count <= min_tokens:
            return start

        saturation_point = min_tokens * _LAMBDA_SATURATION_MULTIPLIER
        if self.token_count >= saturation_point:
            return cap

        progress = (self.token_count - min_tokens) / (saturation_point - min_tokens)
        return start + progress * (cap - start)

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"sentences": self.sentences, "token_count": self.token_count}, f)

    @classmethod
    def load(cls, path: str) -> "UserProfile":
        with open(path, "rb") as f:
            data = pickle.load(f)
        profile = cls()
        profile.sentences = data["sentences"]
        profile.token_count = data["token_count"]
        profile._dirty = True  # refit lazily on first query
        return profile
