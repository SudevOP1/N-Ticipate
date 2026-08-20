"""Phase 4 — hidden Markov model part-of-speech tagger, written from scratch.

The model is the standard bigram HMM over tags:

* an **initial distribution** ``P(t_1)`` — how often each tag starts a sentence;
* a **transition matrix** ``P(t_i | t_i-1)`` — including a final transition to
  an implicit end-of-sentence state, so a tag that never ends a sentence is
  penalised for trying;
* an **emission matrix** ``P(w_i | t_i)``.

All three are Laplace-smoothed with the same ``k``, and decoding is
**log-space Viterbi**.

A word on that ``k``. Add-*one* is the textbook value and it is measurably the
wrong one here: with a vocabulary of ~11k types, ``k = 1`` puts more mass on
the smoothing term than on the counts, flattens every emission row, and drops
the tagger *below* the most-frequent-tag baseline. ``hmm.smoothing_k`` is
therefore swept in the notebook and shipped at ``0.01``, which is on the
plateau. The finding is worth more than the default.

The linear-space Viterbi is implemented too
(:meth:`HMMTagger.trellis` with ``log_space=False``) — not because anything
uses it, but because it underflows to exactly zero at about 125 tokens of
newspaper text, at which point every path scores 0.0 and ``argmax`` starts
returning index 0 for the whole sentence. That is a confidently wrong answer
rather than a degraded one, and it is the cleanest available demonstration of
why log space is not optional. :attr:`Trellis.underflowed` reports it.

``nltk.HiddenMarkovModelTrainer`` appears once, in :func:`nltk_cross_check`,
purely as an independent check that the from-scratch numbers land in the right
neighbourhood. It is never on the prediction path.

Unseen words are the whole difficulty in tagging — on the Penn Treebank they
are ~7% of test tokens and they take a wildly disproportionate share of the
errors, because a smoothed emission row says almost nothing useful about a
word it has never seen. Two orthogonal knobs handle them.

**The prior** (``hmm.unknown_prior``) — what an unseen word looks like before
its spelling is considered:

``hapax``
    Estimate ``P(tag | unseen)`` from the training words that occurred exactly
    once, then divide out ``P(tag)`` to get an emission. Words seen once are
    the best available sample of words seen zero times, and this is the single
    largest win in the phase.
``laplace``
    Whatever the smoothing term alone implies. Honest, weak, and the row the
    ablation table needs for comparison.

**The strategy** (``hmm.unknown_word_strategy``) — what the spelling adds on
top of that prior:

``uniform``
    Nothing. The prior stands as it is.
``most_frequent_tag``
    Everything unknown is pushed towards the corpus's most common tag.
``suffix``
    English morphology: ``-ing``/``-ed`` look like verbs, ``-ly`` like adverbs,
    ``-tion``/``-ness`` like nouns, a mid-sentence capital like a proper noun,
    digits like numerals. The default.

Keeping them orthogonal is what makes the notebook's ablation readable: a 2x3
grid attributing the gain to the prior and to the morphology separately,
rather than one number that moved for two reasons.

The suffix table is deliberately written as *data* (:data:`SUFFIX_RULES`) with
candidate tag names rather than one hardcoded tagset, so Phase 5 can bolt a
Devanagari branch on to the same class without touching a line of the decoder.

Phase 5 is that branch, and it really is only a branch: the Hindi tagger is
this class, these matrices and this decoder, counting a different corpus. Three
things are language-specific and all three sit outside the model —
:data:`DEVANAGARI_SUFFIX_RULES` (Hindi is a suffixing language, so the same
*kind* of rule works with different strings), the shared punctuation and
numeral tag lists which grew the Bureau of Indian Standards spellings
(``PUNC``, ``QFNUM``) alongside the universal ones, and
:func:`clean_tagged_sentences`, which NFC-normalises Devanagari, drops the
blank-tagged tokens NLTK's Hindi corpus ships with, and splits a danda (``।``)
off the word it is glued to.

One English rule has no Hindi counterpart at all: "a capital letter
mid-sentence means proper noun" is the most productive unknown-word rule there
is, and Devanagari is unicameral, so Hindi loses it with nothing to put in its
place. That, plus an OOV rate roughly three times English's on a corpus a
twentieth the size, is where the accuracy gap comes from — measured in notebook
05 rather than asserted.
"""

from __future__ import annotations

import math
import pickle
import random
import unicodedata
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from nticipate.config import get, resolve_path
from nticipate.preprocess import contains_devanagari

#: A sentence of ``(word, tag)`` pairs — the input and output currency here.
TaggedSentence = list[tuple[str, str]]

UNKNOWN_STRATEGIES = ("suffix", "uniform", "most_frequent_tag")
UNKNOWN_PRIORS = ("hapax", "laplace")

#: Stand-in for log(0). Not ``-math.inf``: NumPy warns on ``inf - inf`` inside
#: the Viterbi max, and a very negative finite number ranks identically.
NEG_INF = -1e30


# --------------------------------------------------------------------------
# Unknown-word heuristics
# --------------------------------------------------------------------------
#
# Each rule lists candidate tag names in preference order; whichever names
# exist in the trained tagset are used and the rest ignored. That is what lets
# one rule table serve the universal tagset (``VERB``), the Penn tagset
# (``VBG``) and Phase 5's Hindi tagset with no mapping layer in between.

#: ``(suffix, candidate tags)``, matched longest-first on the lowercased word.
SUFFIX_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ing", ("VBG", "VERB")),
    ("ed", ("VBD", "VERB")),
    ("tion", ("NN", "NOUN")),
    ("ment", ("NN", "NOUN")),
    ("ness", ("NN", "NOUN")),
    ("ship", ("NN", "NOUN")),
    ("ity", ("NN", "NOUN")),
    ("ism", ("NN", "NOUN")),
    ("ist", ("NN", "NOUN")),
    ("ous", ("JJ", "ADJ")),
    ("ive", ("JJ", "ADJ")),
    ("able", ("JJ", "ADJ")),
    ("ible", ("JJ", "ADJ")),
    ("ful", ("JJ", "ADJ")),
    ("less", ("JJ", "ADJ")),
    ("ly", ("RB", "ADV")),
    ("er", ("NN", "NOUN")),
    ("or", ("NN", "NOUN")),
    ("s", ("NNS", "NOUN")),
)

#: Tag *names* for the shape-based rules, across every tagset in play: the
#: universal one, Penn, and the Bureau of Indian Standards tagset NLTK's Hindi
#: corpus uses (``PUNC``, ``QFNUM``). Only the names that exist in the trained
#: tagset are ever used, so listing all of them costs nothing and is what lets
#: one rule table serve three tagsets.
PUNCT_TAGS = (".", "PUNCT", "PUNC", "SYM", "RD_PUNC")
NUM_TAGS = ("CD", "NUM", "QC", "QFNUM")
PROPER_TAGS = ("NNP", "PROPN", "NNPS", "NOUN")

#: Devanagari suffixes, mined from the *training* split in notebook 05 and kept
#: only where they were both frequent and reasonably pure, then cross-checked
#: against Hindi morphology. Same shape as :data:`SUFFIX_RULES`, and read by
#: the same matcher.
#:
#: The oblique-plural ``-ों`` family is the standout: it is unambiguous (94% of
#: training types carrying it are common nouns, and every ``-यों``/``-ियों``
#: type is) and it is productive, which is exactly the combination a rule for
#: *unseen* words needs. Verb endings are next — ``-ने`` for the oblique
#: infinitive, ``-कर`` for the conjunctive participle, ``-ेंगे``/``-ंगे`` for
#: the future — and they are the same tense-and-agreement morphology English
#: reaches for with ``-ing`` and ``-ed``.
#:
#: Deliberately absent: ``-ना`` (bare infinitive, 26% pure — it is also a
#: common noun ending), ``-ता`` (splits between agentive nouns and imperfective
#: verbs, 56%), ``-ार``/``-ान`` (noun-ish but only just, and mostly on proper
#: nouns no spelling rule can reach), and bare ``-गे`` (50% on two training
#: types, and ``-ंगे`` already covers the useful cases). A rule that fires on a
#: coin-flip is worse than no rule: it spends the heuristic's confidence on
#: noise.
#:
#: ``-ीय`` is the one judgement call, kept at 56% train purity. Its misses are
#: almost all ``NNC``, the tag for a non-final element of a compound noun —
#: which is a statement about the word's position, not about its shape, and no
#: suffix rule can predict position. Against the lexical tags it competes with,
#: the rule is right.
DEVANAGARI_SUFFIX_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ियों", ("NN", "NOUN")),
    ("यों", ("NN", "NOUN")),
    ("ेंगे", ("VFM", "VERB")),
    ("ंगे", ("VFM", "VERB")),
    ("ओं", ("NN", "NOUN")),
    ("ों", ("NN", "NOUN")),
    ("ने", ("VNN", "VERB")),
    ("कर", ("VRB", "VERB")),
    ("ीय", ("JJ", "ADJ")),
)

#: Share of the unseen-word probability mass handed to the heuristic's tags.
#: The remaining 1 - this is left on the Laplace floor, so a heuristic that
#: fires wrongly costs the tagger a factor of ten, not the whole sentence.
SUFFIX_CONFIDENCE = 0.9


def candidate_tags(word: str, is_first: bool = False) -> tuple[str, ...]:
    """Guess the tag family of an unseen word from its shape and suffix.

    Returns candidate tag *names* — several spellings of the same idea — or an
    empty tuple when nothing fires, which happens often and simply leaves the
    word sitting on the smoothing floor.

    ``is_first`` suppresses the capitalised-word rule at sentence start, where
    capitalisation carries no information at all.

    Script decides which suffix table applies: a word containing Devanagari is
    matched against :data:`DEVANAGARI_SUFFIX_RULES`, anything else against
    :data:`SUFFIX_RULES`. The shape rules above the split — punctuation,
    digits — are script-independent and shared.
    """
    if not word:
        return ()
    if not any(ch.isalnum() for ch in word):
        return PUNCT_TAGS
    if any(ch.isdigit() for ch in word):
        # Devanagari digits (०-९) are category Nd, so this catches them too.
        return NUM_TAGS

    if contains_devanagari(word):
        # No capitalisation branch above this one, because there is no
        # capitalisation: Devanagari is unicameral. Hindi gets suffixes only.
        return _match_suffix(word, _DEVANAGARI_RULES_BY_LENGTH)

    if not is_first and word[:1].isupper():
        return PROPER_TAGS

    return _match_suffix(word.lower(), _SUFFIX_RULES_BY_LENGTH)


def _match_suffix(
    word: str,
    rules: Sequence[tuple[str, tuple[str, ...]]],
) -> tuple[str, ...]:
    """First matching rule from a longest-suffix-first table."""
    for suffix, tags in rules:
        # Require at least two characters of stem, so "s" does not fire on
        # "is" and "ly" does not fire on "ly". Devanagari matras count as
        # characters here, which is the right unit: "ों" is two code points of
        # ending, and demanding a two-code-point stem keeps the rule off
        # two-letter function words.
        if word.endswith(suffix) and len(word) >= len(suffix) + 2:
            return tags
    return ()


def _by_length(rules) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return tuple(sorted(rules, key=lambda rule: -len(rule[0])))


_SUFFIX_RULES_BY_LENGTH = _by_length(SUFFIX_RULES)
_DEVANAGARI_RULES_BY_LENGTH = _by_length(DEVANAGARI_SUFFIX_RULES)


# --------------------------------------------------------------------------
# Result containers
# --------------------------------------------------------------------------

@dataclass
class Trellis:
    """A full Viterbi lattice, kept so the notebook can print one by hand."""

    tokens: list[str]
    tags: list[str]
    scores: np.ndarray          # (n_tags, n_tokens)
    backpointers: np.ndarray    # (n_tags, n_tokens), int; column 0 is -1
    path: list[str]
    log_space: bool
    underflowed: bool
    final: np.ndarray = field(repr=False, default=None)  # type: ignore[assignment]

    @property
    def best_score(self) -> float:
        """Score of the winning path, including the transition to ``</s>``."""
        return float(self.final.max()) if self.final.size else 0.0

    def column(self, index: int) -> dict[str, float]:
        """Column ``index`` of the lattice as ``{tag: score}``, best first."""
        pairs = {t: float(self.scores[i, index]) for i, t in enumerate(self.tags)}
        return dict(sorted(pairs.items(), key=lambda kv: -kv[1]))


@dataclass
class TaggerEval:
    """Held-out accuracy, split by whether the word was seen in training."""

    tokens: int
    correct: int
    known_tokens: int
    known_correct: int
    unknown_tokens: int
    unknown_correct: int
    per_tag: dict[str, tuple[int, int]]   # tag -> (correct, gold total)

    @property
    def accuracy(self) -> float:
        return self.correct / self.tokens if self.tokens else 0.0

    @property
    def known_accuracy(self) -> float:
        return self.known_correct / self.known_tokens if self.known_tokens else 0.0

    @property
    def unknown_accuracy(self) -> float:
        if not self.unknown_tokens:
            return 0.0
        return self.unknown_correct / self.unknown_tokens

    @property
    def oov_rate(self) -> float:
        """Share of test tokens the model never saw in training.

        Phase 5 leans on this number: the English/Hindi accuracy gap is mostly
        OOV rate, not anything about the languages themselves.
        """
        return self.unknown_tokens / self.tokens if self.tokens else 0.0

    def tag_accuracy(self) -> dict[str, float]:
        return {
            tag: (correct / total if total else 0.0)
            for tag, (correct, total) in sorted(self.per_tag.items())
        }

    def as_dict(self) -> dict:
        return {
            "tokens": self.tokens,
            "accuracy": self.accuracy,
            "known_accuracy": self.known_accuracy,
            "unknown_accuracy": self.unknown_accuracy,
            "oov_rate": self.oov_rate,
        }


# --------------------------------------------------------------------------
# The tagger
# --------------------------------------------------------------------------

class HMMTagger:
    """A bigram HMM tagger with Laplace smoothing and Viterbi decoding.

    Nothing in this class is English-specific. Phase 5 trains the same class on
    a Hindi corpus with no code changes; only :data:`SUFFIX_RULES` grows a
    Devanagari branch.
    """

    def __init__(
        self,
        smoothing_k: float | None = None,
        unknown_strategy: str | None = None,
        use_log_space: bool | None = None,
        tagset: str | None = None,
        unknown_prior: str | None = None,
    ) -> None:
        self.smoothing_k = (
            float(get("hmm.smoothing_k", 0.01)) if smoothing_k is None
            else float(smoothing_k)
        )
        self.unknown_strategy = (
            unknown_strategy if unknown_strategy is not None
            else get("hmm.unknown_word_strategy", "suffix")
        )
        self.unknown_prior = (
            unknown_prior if unknown_prior is not None
            else get("hmm.unknown_prior", "hapax")
        )
        self.use_log_space = (
            bool(get("hmm.use_log_space", True)) if use_log_space is None
            else bool(use_log_space)
        )
        self.tagset = tagset if tagset is not None else get("hmm.tagset", "universal")

        if self.smoothing_k <= 0:
            # k = 0 is MLE, and an MLE emission row gives probability zero to
            # every unseen word, making any sentence containing one undecodable.
            # Phase 2 already documents that failure mode where it belongs.
            raise ValueError(f"smoothing_k must be > 0, got {self.smoothing_k}")
        if self.unknown_strategy not in UNKNOWN_STRATEGIES:
            raise ValueError(
                f"Unknown unknown_word_strategy {self.unknown_strategy!r}; "
                f"expected one of {UNKNOWN_STRATEGIES}"
            )
        if self.unknown_prior not in UNKNOWN_PRIORS:
            raise ValueError(
                f"Unknown unknown_prior {self.unknown_prior!r}; "
                f"expected one of {UNKNOWN_PRIORS}"
            )

        self.tags: list[str] = []
        self.words: list[str] = []
        self._tag_index: dict[str, int] = {}
        self._word_index: dict[str, int] = {}
        self.log_initial: np.ndarray = np.zeros(0)
        self.log_transition: np.ndarray = np.zeros((0, 0))
        self.log_final: np.ndarray = np.zeros(0)
        self.log_emission: np.ndarray = np.zeros((0, 0))
        #: One unseen-word emission vector per prior; see :data:`UNKNOWN_PRIORS`.
        self.unk_priors: dict[str, np.ndarray] = {}
        self.tag_counts: Counter = Counter()
        self.n_sentences = 0
        self._heuristic_cache: dict[tuple[str, bool], np.ndarray] = {}

    # ------------------------------------------------------------- training

    def fit(self, tagged_sentences: Iterable[TaggedSentence]) -> "HMMTagger":
        """Count, smooth, take logs. One pass over the corpus."""
        tag_counts: Counter = Counter()
        start_counts: Counter = Counter()
        final_counts: Counter = Counter()
        trans_counts: defaultdict[str, Counter] = defaultdict(Counter)
        emit_counts: defaultdict[str, Counter] = defaultdict(Counter)
        word_counts: Counter = Counter()
        n_sentences = 0

        for sentence in tagged_sentences:
            if not sentence:
                continue
            n_sentences += 1
            previous: str | None = None
            for position, (word, tag) in enumerate(sentence):
                tag_counts[tag] += 1
                emit_counts[tag][word] += 1
                word_counts[word] += 1
                if position == 0:
                    start_counts[tag] += 1
                else:
                    trans_counts[previous][tag] += 1
                previous = tag
            final_counts[previous] += 1

        if not tag_counts:
            raise ValueError("Cannot fit an HMM on an empty corpus")

        self.tags = sorted(tag_counts)
        self.words = sorted({w for row in emit_counts.values() for w in row})
        self._tag_index = {t: i for i, t in enumerate(self.tags)}
        self._word_index = {w: j for j, w in enumerate(self.words)}
        self.tag_counts = tag_counts
        self.n_sentences = n_sentences
        self._heuristic_cache = {}

        k = self.smoothing_k
        n_tags = len(self.tags)
        n_words = len(self.words)

        # Initial distribution, smoothed over the tags.
        starts = np.array([start_counts[t] for t in self.tags], dtype=np.float64)
        self.log_initial = np.log(starts + k) - math.log(n_sentences + k * n_tags)

        # Transitions. Every tag occurrence is followed by exactly one thing —
        # another tag, or the end of the sentence — so the denominator is that
        # tag's own count, and the +1 in the smoothing term is the end state.
        totals = np.array([tag_counts[t] for t in self.tags], dtype=np.float64)
        denominator = (totals + k * (n_tags + 1)).reshape(-1, 1)

        trans = np.zeros((n_tags, n_tags), dtype=np.float64)
        for previous_tag, row in trans_counts.items():
            i = self._tag_index[previous_tag]
            for tag, count in row.items():
                trans[i, self._tag_index[tag]] = count
        self.log_transition = np.log(trans + k) - np.log(denominator)

        finals = np.array([final_counts[t] for t in self.tags], dtype=np.float64)
        self.log_final = np.log(finals + k) - np.log(denominator.ravel())

        # Emissions. The +1 in this denominator is the unseen-word class, and
        # log_emission_unk is exactly that class's column.
        emit = np.zeros((n_tags, n_words), dtype=np.float64)
        for tag, row in emit_counts.items():
            i = self._tag_index[tag]
            for word, count in row.items():
                emit[i, self._word_index[word]] = count
        emit_denominator = (totals + k * (n_words + 1)).reshape(-1, 1)
        self.log_emission = np.log(emit + k) - np.log(emit_denominator)

        laplace_prior = math.log(k) - np.log(emit_denominator.ravel())
        self.unk_priors = {
            "laplace": laplace_prior,
            "hapax": self._hapax_prior(
                emit_counts, word_counts, totals, laplace_prior
            ),
        }
        return self

    def _hapax_prior(
        self,
        emit_counts: dict[str, Counter],
        word_counts: Counter,
        totals: np.ndarray,
        fallback: np.ndarray,
    ) -> np.ndarray:
        """Unseen-word emission estimated from the words seen exactly once.

        Words with count 1 are the closest observable thing to words with count
        0, so their tag distribution is a far better guess at ``P(tag | unseen)``
        than the smoothing floor is. Viterbi wants an emission, not a posterior,
        so Bayes converts it::

            P(unseen | tag)  ∝  P(tag | unseen) / P(tag)

        The dropped constant ``P(unseen)`` is the same for every tag, and adding
        a constant to a whole trellis column shifts every path through it
        equally — so it cannot change the arg-max, and is not worth computing.
        """
        hapax = np.array(
            [
                sum(1 for word, c in emit_counts[tag].items()
                    if c == 1 and word_counts[word] == 1)
                for tag in self.tags
            ],
            dtype=np.float64,
        )
        if hapax.sum() == 0:
            # No once-seen words at all — only happens on toy corpora. Fall
            # back rather than divide by zero.
            return fallback
        # Add-k again so a tag with no hapax words is unlikely, not impossible.
        k = self.smoothing_k
        p_tag_given_unseen = (hapax + k) / (hapax.sum() + k * len(self.tags))
        p_tag = totals / totals.sum()
        return np.log(p_tag_given_unseen) - np.log(p_tag)

    # ------------------------------------------------------------ inspection

    @property
    def is_fitted(self) -> bool:
        return bool(self.tags)

    @property
    def vocab_size(self) -> int:
        return len(self.words)

    @property
    def log_emission_unk(self) -> np.ndarray:
        """The active unseen-word emission vector, per :attr:`unknown_prior`."""
        return self.unk_priors[self.unknown_prior]

    def reset_cache(self) -> None:
        """Drop memoised unknown-word emissions.

        Call after changing :attr:`unknown_strategy` or :attr:`unknown_prior`
        on a fitted tagger — the ablation helpers do exactly that rather than
        refitting three identical count tables.
        """
        self._heuristic_cache = {}

    def knows(self, word: str) -> bool:
        """Was this word seen in training, case-insensitively?"""
        return word in self._word_index or word.lower() in self._word_index

    def most_frequent_tag(self) -> str:
        return self.tag_counts.most_common(1)[0][0]

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("HMMTagger is not fitted — call fit() first")

    # -------------------------------------------------------------- emission

    def emission_column(self, word: str, is_first: bool = False) -> np.ndarray:
        """Log ``P(word | tag)`` for every tag, as a vector.

        Known words index straight into the emission matrix. Unknown words get
        whatever :attr:`unknown_strategy` says, which is the only place in the
        decoder where anything language-specific happens.
        """
        self._require_fitted()
        column = self._word_index.get(word)
        if column is None:
            # Case backoff before giving up: a sentence-initial "The" is not a
            # new word just because training only ever saw "the" mid-sentence.
            column = self._word_index.get(word.lower())
        if column is not None:
            return self.log_emission[:, column]
        return self._unknown_emission(word, is_first)

    def _unknown_emission(self, word: str, is_first: bool) -> np.ndarray:
        key = (word, is_first)
        cached = self._heuristic_cache.get(key)
        if cached is not None:
            return cached

        base = self.log_emission_unk
        if self.unknown_strategy == "uniform":
            result = base
        elif self.unknown_strategy == "most_frequent_tag":
            result = self._boosted(base, (self.most_frequent_tag(),))
        else:
            result = self._boosted(base, candidate_tags(word, is_first))

        if len(self._heuristic_cache) < 100_000:
            self._heuristic_cache[key] = result
        return result

    def _boosted(self, base: np.ndarray, wanted: Sequence[str]) -> np.ndarray:
        """Interpolate the prior with a point mass on the ``wanted`` tags.

        Concretely: tags the heuristic named keep their prior score, and every
        other tag is pushed down by ``log(1 - SUFFIX_CONFIDENCE)`` — one order
        of magnitude at the default. So a heuristic that fires wrongly costs
        the true tag a factor of ten rather than making it impossible, which
        matters because these rules are wrong a good fraction of the time.

        Only the gaps between tags are meaningful here. Nothing needs to
        normalise across tags: adding a constant to a whole trellis column
        shifts every path through it equally, so it cannot change the arg-max.
        """
        matched = [self._tag_index[t] for t in wanted if t in self._tag_index]
        if not matched:
            return base
        result = base + math.log(1.0 - SUFFIX_CONFIDENCE)
        share = math.log(SUFFIX_CONFIDENCE / len(matched))
        for index in matched:
            result[index] = np.logaddexp(result[index], base[index] + share)
        return result

    # -------------------------------------------------------------- decoding

    def trellis(self, tokens: Sequence[str], log_space: bool | None = None) -> Trellis:
        """Run Viterbi and keep the whole lattice.

        With ``log_space=False`` this is the textbook probability-space
        recursion, kept only so the notebook can watch it underflow.
        """
        self._require_fitted()
        log_space = self.use_log_space if log_space is None else log_space
        tokens = list(tokens)
        n_tags = len(self.tags)

        if not tokens:
            empty = np.zeros((n_tags, 0))
            return Trellis(
                tokens=[], tags=list(self.tags), scores=empty,
                backpointers=empty.astype(np.int64), path=[], log_space=log_space,
                underflowed=False, final=np.zeros(0),
            )

        emissions = [
            self.emission_column(token, is_first=(i == 0))
            for i, token in enumerate(tokens)
        ]

        scores = np.empty((n_tags, len(tokens)), dtype=np.float64)
        backpointers = np.full((n_tags, len(tokens)), -1, dtype=np.int64)

        if log_space:
            scores[:, 0] = self.log_initial + emissions[0]
            for t in range(1, len(tokens)):
                # (previous tag, current tag) matrix, maximised over previous.
                candidates = scores[:, t - 1][:, None] + self.log_transition
                backpointers[:, t] = candidates.argmax(axis=0)
                scores[:, t] = candidates.max(axis=0) + emissions[t]
            final = scores[:, -1] + self.log_final
        else:
            scores[:, 0] = np.exp(self.log_initial + emissions[0])
            transition = np.exp(self.log_transition)
            for t in range(1, len(tokens)):
                candidates = scores[:, t - 1][:, None] * transition
                backpointers[:, t] = candidates.argmax(axis=0)
                scores[:, t] = candidates.max(axis=0) * np.exp(emissions[t])
            final = scores[:, -1] * np.exp(self.log_final)

        # In linear space, a long enough sentence drives every path to exactly
        # 0.0 and the arg-max below becomes meaningless. That is the point.
        underflowed = bool(not log_space and not np.any(final > 0.0))

        best = int(final.argmax())
        reversed_path = [best]
        for t in range(len(tokens) - 1, 0, -1):
            best = int(backpointers[best, t])
            reversed_path.append(best)
        path = [self.tags[i] for i in reversed(reversed_path)]

        return Trellis(
            tokens=tokens, tags=list(self.tags), scores=scores,
            backpointers=backpointers, path=path, log_space=log_space,
            underflowed=underflowed, final=final,
        )

    def viterbi(self, tokens: Sequence[str]) -> list[str]:
        """Most likely tag sequence for ``tokens``."""
        return self.trellis(tokens).path

    def predict_tags(self, tokens: Sequence[str]) -> list[str]:
        """Common interface with :class:`MostFrequentTagBaseline`.

        :func:`evaluate_tagger` calls this, so it does not have to care which
        of the two it was handed.
        """
        return self.viterbi(tokens)

    def tag(self, tokens: Sequence[str]) -> TaggedSentence:
        return list(zip(tokens, self.viterbi(tokens)))

    def sequence_log_probability(self, tagged: TaggedSentence) -> float:
        """Joint log ``P(words, tags)`` — the score Viterbi maximises.

        The tests use it to check by brute force that Viterbi really returns
        the arg-max; the notebook uses it to price one path against another.
        """
        self._require_fitted()
        if not tagged:
            return 0.0
        total = 0.0
        previous: int | None = None
        for position, (word, tag) in enumerate(tagged):
            if tag not in self._tag_index:
                return NEG_INF
            index = self._tag_index[tag]
            if previous is None:
                total += float(self.log_initial[index])
            else:
                total += float(self.log_transition[previous, index])
            total += float(self.emission_column(word, is_first=(position == 0))[index])
            previous = index
        return total + float(self.log_final[previous])

    # ------------------------------------------------------------ evaluation

    def evaluate(self, tagged_sentences: Sequence[TaggedSentence]) -> TaggerEval:
        return evaluate_tagger(self, tagged_sentences)

    # ----------------------------------------------------------- persistence

    def to_dict(self) -> dict:
        return {
            "smoothing_k": self.smoothing_k,
            "unknown_strategy": self.unknown_strategy,
            "unknown_prior": self.unknown_prior,
            "use_log_space": self.use_log_space,
            "tagset": self.tagset,
            "tags": self.tags,
            "words": self.words,
            "log_initial": self.log_initial,
            "log_transition": self.log_transition,
            "log_final": self.log_final,
            "log_emission": self.log_emission,
            "unk_priors": self.unk_priors,
            "tag_counts": dict(self.tag_counts),
            "n_sentences": self.n_sentences,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HMMTagger":
        tagger = cls(
            smoothing_k=data["smoothing_k"],
            unknown_strategy=data["unknown_strategy"],
            use_log_space=data["use_log_space"],
            tagset=data["tagset"],
            unknown_prior=data["unknown_prior"],
        )
        tagger.tags = list(data["tags"])
        tagger.words = list(data["words"])
        tagger._tag_index = {t: i for i, t in enumerate(tagger.tags)}
        tagger._word_index = {w: j for j, w in enumerate(tagger.words)}
        tagger.log_initial = np.asarray(data["log_initial"])
        tagger.log_transition = np.asarray(data["log_transition"])
        tagger.log_final = np.asarray(data["log_final"])
        tagger.log_emission = np.asarray(data["log_emission"])
        tagger.unk_priors = {k: np.asarray(v) for k, v in data["unk_priors"].items()}
        tagger.tag_counts = Counter(data["tag_counts"])
        tagger.n_sentences = data["n_sentences"]
        return tagger

    def save(self, path: str | Path) -> Path:
        target = resolve_path(path, create=True)
        with target.open("wb") as fh:
            pickle.dump(self.to_dict(), fh, protocol=pickle.HIGHEST_PROTOCOL)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "HMMTagger":
        source = resolve_path(path)
        with source.open("rb") as fh:
            return cls.from_dict(pickle.load(fh))

    def __repr__(self) -> str:
        if not self.is_fitted:
            return "HMMTagger(unfitted)"
        return (
            f"HMMTagger(tags={len(self.tags)}, vocab={len(self.words):,}, "
            f"k={self.smoothing_k}, unknown={self.unknown_prior}+"
            f"{self.unknown_strategy})"
        )


# --------------------------------------------------------------------------
# Baseline
# --------------------------------------------------------------------------

class MostFrequentTagBaseline:
    """Assign each word its most frequent training tag. No context at all.

    This is the number the HMM has to beat to have earned its existence, and it
    is a stronger baseline than it sounds — English word/tag ambiguity is
    concentrated in a small set of very frequent words, so most tokens have
    only one plausible tag and context adds nothing.
    """

    def __init__(self) -> None:
        self.word_tag: dict[str, str] = {}
        self.surface_forms: set[str] = set()
        self.default_tag = "NOUN"

    def fit(
        self,
        tagged_sentences: Iterable[TaggedSentence],
    ) -> "MostFrequentTagBaseline":
        counts: defaultdict[str, Counter] = defaultdict(Counter)
        totals: Counter = Counter()
        for sentence in tagged_sentences:
            for word, tag in sentence:
                counts[word.lower()][tag] += 1
                totals[tag] += 1
                self.surface_forms.add(word)
        self.word_tag = {w: row.most_common(1)[0][0] for w, row in counts.items()}
        if totals:
            self.default_tag = totals.most_common(1)[0][0]
        return self

    def knows(self, word: str) -> bool:
        # Deliberately the same definition as :meth:`HMMTagger.knows` rather
        # than a lookup in the lowercased table, so the two taggers report the
        # same OOV rate and their unseen-word columns compare like for like.
        return word in self.surface_forms or word.lower() in self.surface_forms

    def predict_tags(self, tokens: Sequence[str]) -> list[str]:
        return [self.word_tag.get(t.lower(), self.default_tag) for t in tokens]

    def tag(self, tokens: Sequence[str]) -> TaggedSentence:
        return list(zip(tokens, self.predict_tags(tokens)))

    def evaluate(self, tagged_sentences: Sequence[TaggedSentence]) -> TaggerEval:
        return evaluate_tagger(self, tagged_sentences)

    def __repr__(self) -> str:
        return f"MostFrequentTagBaseline(words={len(self.word_tag):,})"


# --------------------------------------------------------------------------
# Evaluation helpers (Phase 4 deliverables)
# --------------------------------------------------------------------------

def evaluate_tagger(tagger, tagged_sentences: Sequence[TaggedSentence]) -> TaggerEval:
    """Token accuracy, split into seen and unseen words.

    The split matters more than the headline number: two taggers with identical
    overall accuracy can be doing completely different things, and the unseen
    column is where the suffix heuristic either earns its place or does not.
    """
    tokens = correct = 0
    known_tokens = known_correct = 0
    unknown_tokens = unknown_correct = 0
    per_tag: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])

    for sentence in tagged_sentences:
        if not sentence:
            continue
        words = [w for w, _ in sentence]
        gold = [t for _, t in sentence]
        predicted = tagger.predict_tags(words)
        for word, gold_tag, predicted_tag in zip(words, gold, predicted):
            hit = int(gold_tag == predicted_tag)
            tokens += 1
            correct += hit
            per_tag[gold_tag][1] += 1
            per_tag[gold_tag][0] += hit
            if tagger.knows(word):
                known_tokens += 1
                known_correct += hit
            else:
                unknown_tokens += 1
                unknown_correct += hit

    return TaggerEval(
        tokens=tokens,
        correct=correct,
        known_tokens=known_tokens,
        known_correct=known_correct,
        unknown_tokens=unknown_tokens,
        unknown_correct=unknown_correct,
        per_tag={tag: (c, n) for tag, (c, n) in per_tag.items()},
    )


def collect_predictions(
    tagger,
    tagged_sentences: Sequence[TaggedSentence],
) -> list[tuple[str, str, str]]:
    """``(word, gold tag, predicted tag)`` for every test token."""
    rows: list[tuple[str, str, str]] = []
    for sentence in tagged_sentences:
        if not sentence:
            continue
        words = [w for w, _ in sentence]
        predicted = tagger.predict_tags(words)
        rows.extend(
            (word, gold, guess)
            for (word, gold), guess in zip(sentence, predicted)
        )
    return rows


def confusion_matrix(
    predictions: Sequence[tuple[str, str, str]],
    tags: Sequence[str] | None = None,
) -> tuple[list[str], np.ndarray]:
    """Rows = gold tag, columns = predicted tag."""
    if tags is None:
        tags = sorted({t for _, gold, guess in predictions for t in (gold, guess)})
    index = {t: i for i, t in enumerate(tags)}
    matrix = np.zeros((len(tags), len(tags)), dtype=np.int64)
    for _, gold, guess in predictions:
        if gold in index and guess in index:
            matrix[index[gold], index[guess]] += 1
    return list(tags), matrix


def confusion_pairs(
    predictions: Sequence[tuple[str, str, str]],
    top: int = 10,
) -> list[tuple[str, str, int]]:
    """The ``top`` most common ``(gold, predicted)`` mistakes, worst first."""
    errors: Counter = Counter(
        (gold, guess) for _, gold, guess in predictions if gold != guess
    )
    return [(gold, guess, n) for (gold, guess), n in errors.most_common(top)]


def nltk_cross_check(
    train: Sequence[TaggedSentence],
    test: Sequence[TaggedSentence],
) -> dict | None:
    """Train NLTK's HMM on the same split, for an independent second opinion.

    Read this number with care, and read it on OOV-free sentences.
    ``HiddenMarkovModelTrainer.train_supervised`` defaults to an essentially
    unsmoothed emission estimator, so on text containing unseen words it does
    not merely lose — it collapses. That says something about NLTK's defaults
    and nothing about whether the implementation in this module is correct.

    The real correctness check on the decoder is in ``tests/test_hmm.py``,
    where Viterbi is compared against brute-force enumeration of every tag
    sequence on short inputs. This function is the weaker, secondary check that
    the from-scratch numbers land in the right neighbourhood.

    Returns ``None`` when NLTK is unavailable. The shipped prediction path
    never calls it, and ``hmm.cross_check_with_nltk`` defaults to false.
    """
    try:
        from nltk.tag import hmm as nltk_hmm
    except ImportError:
        return None

    with warnings.catch_warnings():
        # NLTK's own matrix build overflows on its log(0) entries. Not ours.
        warnings.simplefilter("ignore")
        trainer = nltk_hmm.HiddenMarkovModelTrainer()
        model = trainer.train_supervised([list(s) for s in train])

        tokens = correct = 0
        for sentence in test:
            words = [w for w, _ in sentence]
            try:
                predicted: list[str | None] = [t for _, t in model.tag(words)]
            except ValueError:
                # NLTK's supervised HMM can fail outright on a sentence of
                # entirely unseen words. Counting it wrong is the honest thing.
                predicted = [None] * len(words)
            for (_, gold), guess in zip(sentence, predicted):
                tokens += 1
                correct += int(gold == guess)

    return {
        "tokens": tokens,
        "correct": correct,
        "accuracy": correct / tokens if tokens else 0.0,
    }


def without_oov(
    tagger,
    sentences: Sequence[TaggedSentence],
    limit: int | None = None,
) -> list[TaggedSentence]:
    """Sentences in which every word was seen in training.

    Comparing two taggers on these isolates the decoder from the unknown-word
    policy, which is otherwise the only thing the comparison measures.
    """
    kept = [s for s in sentences if s and all(tagger.knows(w) for w, _ in s)]
    return kept[:limit] if limit else kept


# --------------------------------------------------------------------------
# Corpus loading
# --------------------------------------------------------------------------

#: Corpora whose NLTK reader needs a file id (Phase 5's Hindi lives here).
NLTK_FILEIDS = {"indian": "hindi.pos"}

#: The Devanagari full stop and its doubled form — sentence punctuation, not
#: part of the word in front of them.
DANDA = "।॥"


def _punctuation_tag(sentences: Sequence[TaggedSentence]) -> str:
    """Whatever tag *this* corpus gives standalone punctuation.

    Read off the corpus rather than hardcoded, because the answer is ``.`` in
    the universal tagset and ``PUNC`` in the Hindi one, and a danda split out
    of a word has to be tagged in the same currency as the rest of the file.
    """
    counts = Counter(
        tag for sentence in sentences for word, tag in sentence
        if tag.strip() and word.strip() and not any(ch.isalnum() for ch in word)
    )
    return counts.most_common(1)[0][0] if counts else PUNCT_TAGS[0]


def clean_tagged_sentences(
    sentences: Iterable[TaggedSentence],
    normalize: str | None = None,
) -> list[TaggedSentence]:
    """Three repairs a gold corpus should not need, and the Hindi one does.

    **NFC.** Devanagari spells several consonants two ways — क़ as one code
    point, or क plus a nukta as two. Un-normalised they are different strings,
    so they are different word types, so the vocabulary and the OOV rate both
    inflate for no linguistic reason. ``preprocessing.normalize_unicode``
    already fixes this for Phase 1's text; this does it for tagged input.

    **Blank tags.** ``hindi.pos`` carries a couple of dozen tokens whose tag
    column is empty. Left in, ``""`` becomes a tag like any other: it takes a
    row in the transition matrix, a row in the emission matrix, and it is
    unscoreable — no prediction can ever be right.

    **Danda.** ``।`` glued to the word before it invents a new word type out of
    every sentence-final word. Split off it is one ordinary punctuation token,
    tagged as :func:`_punctuation_tag` says this corpus tags punctuation.

    English input passes through unchanged, which is the point: the loader
    calls this for every language and there is no branch on which one.
    """
    if normalize is None:
        normalize = get("preprocessing.normalize_unicode", "NFC")

    materialised = [list(sentence) for sentence in sentences]
    punct_tag = _punctuation_tag(materialised)

    cleaned: list[TaggedSentence] = []
    for sentence in materialised:
        row: TaggedSentence = []
        for word, tag in sentence:
            if normalize:
                word = unicodedata.normalize(normalize, word)
            word, tag = word.strip(), tag.strip()
            if not word or not tag:
                continue
            stem = word.rstrip(DANDA)
            if stem:
                row.append((stem, tag))
                row.extend((mark, punct_tag) for mark in word[len(stem):])
            else:
                # The token is nothing but dandas — punctuation already.
                row.extend((mark, punct_tag) for mark in word)
        if row:
            cleaned.append(row)
    return cleaned


def read_conll(path: str | Path) -> list[TaggedSentence]:
    """Read a two-column ``word<TAB>tag`` file, blank line between sentences.

    The simplest interchange format that survives a text editor, which matters
    for Phase 5 where the Hindi corpus may have to be assembled by hand.
    """
    source = resolve_path(path)
    sentences: list[TaggedSentence] = []
    current: TaggedSentence = []
    with source.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line.strip():
                if current:
                    sentences.append(current)
                    current = []
                continue
            if line.startswith("#"):
                continue
            parts = line.split("\t") if "\t" in line else line.rsplit(" ", 1)
            if len(parts) != 2:
                continue
            current.append((parts[0], parts[1]))
    if current:
        sentences.append(current)
    return sentences


def write_conll(path: str | Path, sentences: Iterable[TaggedSentence]) -> Path:
    target = resolve_path(path, create=True)
    with target.open("w", encoding="utf-8") as fh:
        for sentence in sentences:
            for word, tag in sentence:
                fh.write(f"{word}\t{tag}\n")
            fh.write("\n")
    return target


def _load_nltk_tagged(
    name: str,
    tagset: str | None,
    limit: int | None,
) -> list[TaggedSentence]:
    corpus = getattr(__import__("nltk.corpus", fromlist=[name]), name)
    fileids = NLTK_FILEIDS.get(name)
    # The universal mapping exists for some corpora and not others; ask for it
    # first, fall back to the corpus's native tagset rather than failing.
    attempts: list[dict] = []
    if tagset and tagset != "penn":
        attempts.append({"tagset": tagset})
    attempts.append({})

    last: Exception | None = None
    for kwargs in attempts:
        try:
            sentences = (
                corpus.tagged_sents(fileids, **kwargs) if fileids
                else corpus.tagged_sents(**kwargs)
            )
            selected = sentences[:limit] if limit else sentences
            return [[(w, t) for w, t in sentence] for sentence in selected]
        except Exception as exc:   # unmapped tagset, corpus not downloaded, ...
            last = exc
    raise RuntimeError(f"NLTK corpus {name!r} unusable: {last}")


def load_tagged_sentences(
    language: str | None = None,
    limit: int | None = None,
) -> list[TaggedSentence]:
    """Load a tagged corpus for ``language`` (``english`` or ``hindi``).

    Prefers the NLTK corpus named by ``hmm.corpora.<language>_nltk`` — it is
    gold-tagged and large — and falls back to the ``.conll`` file at
    ``hmm.corpora.<language>``, so a fresh clone with nothing downloaded still
    runs. Phase 5 changes the argument, not the code.

    Either way the result goes through :func:`clean_tagged_sentences` — the
    Hindi corpus needs it and English is unaffected by it.
    """
    if language is None:
        language = get("hmm.language", "english")
    tagset = get("hmm.tagset", "universal")

    nltk_name = get(f"hmm.corpora.{language}_nltk")
    if nltk_name:
        try:
            return clean_tagged_sentences(_load_nltk_tagged(nltk_name, tagset, limit))
        except Exception:
            pass

    fallback = get(f"hmm.corpora.{language}")
    if fallback:
        path = resolve_path(fallback)
        if path.is_file():
            sentences = read_conll(path)
            return clean_tagged_sentences(sentences[:limit] if limit else sentences)

    raise FileNotFoundError(
        f"No tagged corpus for {language!r}: NLTK corpus {nltk_name!r} is not "
        f"available and {fallback!r} does not exist. Run: python setup_env.py"
    )


def train_test_split_tagged(
    sentences: Sequence[TaggedSentence],
    train: float | None = None,
    seed: int | None = None,
) -> tuple[list[TaggedSentence], list[TaggedSentence]]:
    """Shuffle at sentence level and split, using ``hmm.split``.

    A separate ratio and seed from Phase 1's split on purpose: this is a
    different corpus doing a different task, and sharing the split would imply
    a relationship between the two that does not exist.
    """
    if train is None:
        train = get("hmm.split.train", 0.8)
    if seed is None:
        seed = get("hmm.split.seed", 42)
    if not 0.0 < train < 1.0:
        raise ValueError(f"train fraction must be in (0, 1), got {train}")

    items = [list(s) for s in sentences]
    random.Random(seed).shuffle(items)
    cut = int(len(items) * train)
    return items[:cut], items[cut:]


def train_tagger(
    sentences: Iterable[TaggedSentence] | None = None,
    **kwargs,
) -> HMMTagger:
    """Fit a tagger, loading the configured corpus when none is given."""
    if sentences is None:
        sentences = load_tagged_sentences()
    return HMMTagger(**kwargs).fit(sentences)


def unknown_word_ablation(
    train: Sequence[TaggedSentence],
    test: Sequence[TaggedSentence],
    priors: Sequence[str] = UNKNOWN_PRIORS,
    strategies: Sequence[str] = UNKNOWN_STRATEGIES,
) -> list[dict]:
    """Accuracy for every (prior, strategy) pair — the Phase 4 ablation.

    Counting happens once and the two knobs are swapped on the fitted model:
    every cell of this grid shares identical transition and emission matrices
    and differs only in what happens to a word outside the vocabulary. Refitting
    per cell would multiply the runtime for no change in the counts.
    """
    tagger = HMMTagger().fit(train)
    rows = []
    for prior in priors:
        for strategy in strategies:
            tagger.unknown_prior = prior
            tagger.unknown_strategy = strategy
            tagger.reset_cache()
            result = tagger.evaluate(test)
            rows.append({
                "prior": prior,
                "strategy": strategy,
                "accuracy": result.accuracy,
                "known_accuracy": result.known_accuracy,
                "unknown_accuracy": result.unknown_accuracy,
            })
    return rows


def smoothing_sweep(
    train: Sequence[TaggedSentence],
    test: Sequence[TaggedSentence],
    values: Sequence[float] = (1.0, 0.1, 0.01, 0.001, 1e-4, 1e-5),
) -> list[dict]:
    """Accuracy against the Laplace ``k`` — why add-one is the wrong default.

    Refits per value, unlike the ablation above, because ``k`` is baked into
    the counts.
    """
    rows = []
    for k in values:
        result = HMMTagger(smoothing_k=k).fit(train).evaluate(test)
        rows.append({
            "k": k,
            "accuracy": result.accuracy,
            "known_accuracy": result.known_accuracy,
            "unknown_accuracy": result.unknown_accuracy,
        })
    return rows


# --------------------------------------------------------------------------
# Phase 5 — cross-language comparison
# --------------------------------------------------------------------------

def tagged_corpus_stats(sentences: Sequence[TaggedSentence]) -> dict:
    """Shape of a tagged corpus: size, type-token ratio, tagset.

    The type-token ratio is the number worth reading next to an accuracy. A
    morphologically rich language spells one lemma many ways, so it burns
    through vocabulary faster at equal token count — a higher TTR on a smaller
    corpus predicts the OOV rate, and the OOV rate predicts most of the gap
    between two taggers that are otherwise the same code.
    """
    tokens = [word for sentence in sentences for word, _ in sentence]
    tags = Counter(tag for sentence in sentences for _, tag in sentence)
    types = set(tokens)
    return {
        "sentences": len(sentences),
        "tokens": len(tokens),
        "types": len(types),
        "ttr": len(types) / len(tokens) if tokens else 0.0,
        "mean_sentence_len": len(tokens) / len(sentences) if sentences else 0.0,
        "tags": len(tags),
        "most_common_tag": tags.most_common(1)[0][0] if tags else "",
    }


def suffix_rule_report(
    sentences: Sequence[TaggedSentence],
    rules: Sequence[tuple[str, tuple[str, ...]]] | None = None,
) -> list[dict]:
    """Per-rule accuracy of the suffix table, counted over word *types*.

    Types rather than tokens, because these rules only ever fire on words the
    model has not seen; counting a frequent word once per occurrence would
    measure a population the heuristic never meets.

    ``rules`` defaults to whichever table the word's script selects, and the
    first-match order is the tagger's own, so a rule shadowed by a longer one
    shows up here with the smaller support it actually has.
    """
    types: defaultdict[str, Counter] = defaultdict(Counter)
    for sentence in sentences:
        for word, tag in sentence:
            types[word][tag] += 1

    fired: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])
    labels: dict[str, tuple[str, ...]] = {}
    for word, tag_counts in types.items():
        table = rules
        if table is None:
            table = (
                _DEVANAGARI_RULES_BY_LENGTH if contains_devanagari(word)
                else _SUFFIX_RULES_BY_LENGTH
            )
        for suffix, tags in table:
            if word.endswith(suffix) and len(word) >= len(suffix) + 2:
                gold = tag_counts.most_common(1)[0][0]
                labels[suffix] = tags
                fired[suffix][1] += 1
                fired[suffix][0] += int(gold in tags)
                break

    rows = [
        {
            "suffix": suffix,
            "tags": labels[suffix],
            "types": total,
            "correct": hits,
            "purity": hits / total if total else 0.0,
        }
        for suffix, (hits, total) in fired.items()
    ]
    return sorted(rows, key=lambda row: -row["types"])


def language_comparison(
    languages: Sequence[str] = ("english", "hindi"),
    limit: int | None = None,
    **tagger_kwargs,
) -> list[dict]:
    """Train and score the same class on each language — the Phase 5 table.

    One row per language, carrying the corpus shape, the tagger, the
    most-frequent-tag baseline and the OOV rate together, because reading any
    of them alone invites the wrong conclusion: Hindi scoring below English
    says nothing until the OOV rates sit in the next column.

    The fitted tagger and its held-out split ride along under ``tagger`` and
    ``test`` so the caller can carry on to a confusion matrix without paying
    for a second fit.
    """
    rows = []
    for language in languages:
        sentences = load_tagged_sentences(language, limit=limit)
        train, test = train_test_split_tagged(sentences)
        tagger = HMMTagger(**tagger_kwargs).fit(train)
        result = tagger.evaluate(test)
        baseline = MostFrequentTagBaseline().fit(train).evaluate(test)
        rows.append({
            "language": language,
            **tagged_corpus_stats(sentences),
            "train_sentences": len(train),
            "test_sentences": len(test),
            "accuracy": result.accuracy,
            "known_accuracy": result.known_accuracy,
            "unknown_accuracy": result.unknown_accuracy,
            "oov_rate": result.oov_rate,
            "baseline_accuracy": baseline.accuracy,
            "tagger": tagger,
            "test": test,
        })
    return rows
