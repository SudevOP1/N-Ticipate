"""
Phases 4 & 5: HMM part-of-speech tagger.

Same class serves both experiments -- only the training corpus differs
(CFG['hmm']['english'] vs CFG['hmm']['regional']). That reuse is itself
worth a line in the report: the model doesn't care what language it's
counting, only the corpus statistics change.

Trained from scratch (transition matrix A, emission matrix B, initial
distribution pi) with Viterbi decoding in log space -- no
nltk.HiddenMarkovModelTrainer in the shipped path, only as a
correctness cross-check in notebooks/04_hmm_english.ipynb.

Unlike the n-gram model (nticipate/ngram.py), tokens here keep their
original casing -- capitalization is itself a POS signal (proper
nouns), so this module never routes through preprocess.py's
counting-key lowercasing.
"""

import math
from collections import defaultdict, Counter

from nticipate.config import load_config

CFG = load_config()

_DEVANAGARI_LOW, _DEVANAGARI_HIGH = 0x0900, 0x097F


def _is_devanagari(word: str) -> bool:
    return any(_DEVANAGARI_LOW <= ord(ch) <= _DEVANAGARI_HIGH for ch in word)


class HMMTagger:
    def __init__(self, unknown_word_strategy: str | None = None):
        self.unknown_word_strategy = unknown_word_strategy or CFG["hmm"]["unknown_word_strategy"]
        self.tags: set[str] = set()
        self.vocab: set[str] = set()
        self.transition: dict[str, Counter] = defaultdict(Counter)   # tag_{i-1} -> Counter(tag_i)
        self.emission: dict[str, Counter] = defaultdict(Counter)      # tag -> Counter(word)
        self.initial: Counter = Counter()                             # tag -> count as sentence-initial

        # cached row totals, built by fit() -- avoids re-summing Counters
        # on every _log_transition / _log_emission call during Viterbi
        self.transition_totals: dict[str, int] = {}
        self.emission_totals: dict[str, int] = {}

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    def fit(self, tagged_sentences: list[list[tuple[str, str]]]) -> "HMMTagger":
        """Count transitions, emissions, and initial tag frequencies."""
        self.transition = defaultdict(Counter)
        self.emission = defaultdict(Counter)
        self.initial = Counter()
        self.tags = set()
        self.vocab = set()

        for sentence in tagged_sentences:
            if not sentence:
                continue
            prev_tag = None
            for i, (word, tag) in enumerate(sentence):
                self.tags.add(tag)
                self.vocab.add(word)
                self.emission[tag][word] += 1
                if i == 0:
                    self.initial[tag] += 1
                else:
                    self.transition[prev_tag][tag] += 1
                prev_tag = tag

        self.transition_totals = {tag: sum(counter.values()) for tag, counter in self.transition.items()}
        self.emission_totals = {tag: sum(counter.values()) for tag, counter in self.emission.items()}
        return self

    # ------------------------------------------------------------------
    # smoothed log-probabilities
    # ------------------------------------------------------------------
    def _laplace_k(self) -> float:
        return CFG["hmm"]["laplace_smoothing_emissions"]

    def _log_initial(self, tag: str) -> float:
        k = self._laplace_k()
        total = sum(self.initial.values())
        count = self.initial.get(tag, 0)
        num_tags = max(len(self.tags), 1)
        prob = (count + k) / (total + k * num_tags)
        return math.log(prob) if prob > 0 else float("-inf")

    def _log_transition(self, prev_tag: str, tag: str) -> float:
        k = self._laplace_k()
        total = self.transition_totals.get(prev_tag, 0)
        count = self.transition.get(prev_tag, {}).get(tag, 0)
        num_tags = max(len(self.tags), 1)
        prob = (count + k) / (total + k * num_tags)
        return math.log(prob) if prob > 0 else float("-inf")

    def _log_emission(self, tag: str, word: str) -> float:
        """Falls back to self._guess_tag_by_suffix for OOV words.

        Known words: standard add-k smoothed MLE over the vocabulary.
        Unknown words: rather than a proper statistical unknown-word
        model (e.g. Good-Turing), probability mass is split 50/50
        between the suffix-guessed tag and everything else -- a common,
        simple approximation for a from-scratch HMM. See report/notes.md.
        """
        if word in self.vocab:
            k = self._laplace_k()
            total = self.emission_totals.get(tag, 0)
            count = self.emission.get(tag, {}).get(word, 0)
            v = max(len(self.vocab), 1)
            prob = (count + k) / (total + k * v)
            return math.log(prob) if prob > 0 else float("-inf")

        guessed_tag = self._guess_tag_by_suffix(word)
        num_tags = max(len(self.tags), 1)
        if tag == guessed_tag:
            prob = 0.5
        else:
            prob = 0.5 / max(num_tags - 1, 1)
        return math.log(prob)

    # ------------------------------------------------------------------
    # unknown-word handling
    # ------------------------------------------------------------------
    def _most_common_tag(self) -> str:
        if self.emission_totals:
            return max(self.emission_totals, key=self.emission_totals.get)
        return next(iter(self.tags), "NOUN")

    def _guess_tag_by_suffix(self, word: str) -> str:
        """-ing -> VERB, -ly -> ADV, -tion -> NOUN, capitalized -> proper-noun
        tag, etc. Routes to a Devanagari-specific variant when the word
        contains Devanagari characters -- no capitalization signal exists
        there, so Phase 5 leans entirely on suffix morphology instead.
        """
        if _is_devanagari(word):
            return self._guess_tag_by_suffix_devanagari(word)
        return self._guess_tag_by_suffix_english(word)

    def _guess_tag_by_suffix_english(self, word: str) -> str:
        lower = word.lower()

        # Capitalized, alphabetic, not sentence-initial-only-by-accident:
        # treat as a proper noun. Penn tagset has NNP; Universal tagset
        # (the default, CFG['hmm']['english']['tagset']) folds proper
        # nouns into plain NOUN -- use whichever this model was trained on.
        if word[:1].isupper() and word.isalpha():
            if "NNP" in self.tags:
                return "NNP"
            if "NOUN" in self.tags:
                return "NOUN"
            return self._most_common_tag()

        if lower.endswith("ing") or lower.endswith("ed"):
            return "VERB" if "VERB" in self.tags else self._most_common_tag()
        if lower.endswith("ly"):
            return "ADV" if "ADV" in self.tags else self._most_common_tag()
        if lower.endswith(("tion", "ment", "ness", "ity")):
            return "NOUN" if "NOUN" in self.tags else self._most_common_tag()
        if lower.endswith(("able", "ible", "ful", "ous", "ive", "al")):
            return "ADJ" if "ADJ" in self.tags else self._most_common_tag()
        if lower.endswith("s") and len(lower) > 2:
            return "NOUN" if "NOUN" in self.tags else self._most_common_tag()

        return "NOUN" if "NOUN" in self.tags else self._most_common_tag()

    def _guess_tag_by_suffix_devanagari(self, word: str) -> str:
        """First-pass Hindi/Marathi suffix heuristics -- no capitalization
        signal exists in Devanagari, so this leans entirely on common
        verb/noun morphological markers. This sandbox has no network
        access to download NLTK's 'indian' corpus, so these rules are
        untested here; Phase 5's notebook validates and tunes them
        against real Hindi tagged data.
        """
        verb_tag = "VERB" if "VERB" in self.tags else self._most_common_tag()
        noun_tag = "NOUN" if "NOUN" in self.tags else self._most_common_tag()

        # infinitive / habitual / past verb markers
        if word.endswith(("ना", "ता", "ती", "ते", "या", "एं")):
            return verb_tag
        # common noun plural / oblique markers
        if word.endswith(("ों", "ाएं", "ियाँ", "ियों")):
            return noun_tag

        return noun_tag

    # ------------------------------------------------------------------
    # decoding
    # ------------------------------------------------------------------
    def _viterbi_tables(
        self, tokens: list[str]
    ) -> tuple[list[dict[str, float]], list[dict[str, str | None]], list[str]]:
        """Compute the Viterbi dp/backpointer tables. Split out from
        viterbi() so notebooks/04_hmm_english.ipynb can render the full
        trellis (dp values at every position/tag) without duplicating
        the algorithm -- one source of truth for both the shipped
        decoder and the worked example in the report.
        """
        tags_list = sorted(self.tags)
        if not tags_list:
            raise ValueError("HMMTagger.viterbi called before fit() -- no tags known")

        n = len(tokens)
        dp: list[dict[str, float]] = [{} for _ in range(n)]
        backpointer: list[dict[str, str | None]] = [{} for _ in range(n)]

        for tag in tags_list:
            dp[0][tag] = self._log_initial(tag) + self._log_emission(tag, tokens[0])
            backpointer[0][tag] = None

        for i in range(1, n):
            word = tokens[i]
            # emission doesn't depend on prev_tag -- compute once per tag per step
            emission_by_tag = {tag: self._log_emission(tag, word) for tag in tags_list}
            for tag in tags_list:
                best_prev, best_score = None, float("-inf")
                for prev_tag in tags_list:
                    score = dp[i - 1][prev_tag] + self._log_transition(prev_tag, tag)
                    if score > best_score:
                        best_score, best_prev = score, prev_tag
                dp[i][tag] = best_score + emission_by_tag[tag]
                backpointer[i][tag] = best_prev

        return dp, backpointer, tags_list

    def viterbi(self, tokens: list[str]) -> list[str]:
        """Return the most likely tag sequence for `tokens` via log-space Viterbi."""
        if not tokens:
            return []

        dp, backpointer, _ = self._viterbi_tables(tokens)
        n = len(tokens)
        best_final_tag = max(dp[n - 1], key=dp[n - 1].get)
        result: list[str] = [""] * n
        result[n - 1] = best_final_tag
        for i in range(n - 2, -1, -1):
            result[i] = backpointer[i + 1][result[i + 1]]
        return result

    # ------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------
    def accuracy(self, tagged_sentences: list[list[tuple[str, str]]]) -> float:
        """Token-level tagging accuracy on held-out data."""
        correct, total = 0, 0
        for sentence in tagged_sentences:
            if not sentence:
                continue
            tokens = [word for word, _ in sentence]
            gold_tags = [tag for _, tag in sentence]
            predicted_tags = self.viterbi(tokens)
            for gold, predicted in zip(gold_tags, predicted_tags):
                total += 1
                if gold == predicted:
                    correct += 1
        return correct / total if total else 0.0

    def confusion_pairs(self, tagged_sentences: list[list[tuple[str, str]]]) -> Counter:
        """Counter of (gold_tag, predicted_tag) pairs, for a confusion
        matrix / most-confused-pairs table in the report.
        """
        pairs: Counter = Counter()
        for sentence in tagged_sentences:
            if not sentence:
                continue
            tokens = [word for word, _ in sentence]
            gold_tags = [tag for _, tag in sentence]
            predicted_tags = self.viterbi(tokens)
            for gold, predicted in zip(gold_tags, predicted_tags):
                pairs[(gold, predicted)] += 1
        return pairs

    # ------------------------------------------------------------------
    # single-word tag guess (context-free) -- used by Phase 6's reranker
    # ------------------------------------------------------------------
    def most_likely_tag_for_word(self, word: str) -> str:
        """Best single-tag guess for a word, independent of any sentence
        context -- e.g. "the typical tag of 'quickly'" rather than "the
        tag of 'quickly' in this specific sentence" (that's viterbi()'s
        job). Used by Predictor._rerank_with_pos to estimate a candidate
        word's typical POS tag.

        The n-gram model's vocabulary is lowercased
        (preprocess.py's counting-key convention) while this tagger's
        vocabulary keeps original casing (capitalization is itself a POS
        signal) -- so a candidate word is tried as given, then
        title-cased, before falling back to the unknown-word suffix
        heuristic on the original casing.
        """
        candidates_to_try = [word]
        if word and word[0].islower():
            candidates_to_try.append(word[0].upper() + word[1:])

        for candidate in candidates_to_try:
            if candidate in self.vocab:
                return max(self.tags, key=lambda tag: self._log_emission(tag, candidate))

        return self._guess_tag_by_suffix(word)

    # ------------------------------------------------------------------
    # Phase 6 hook
    # ------------------------------------------------------------------
    def next_tag_distribution(self, tag_context: tuple[str, ...]) -> dict[str, float]:
        """P(next_tag | previous tags) -- consumed by Phase 6 reranker.

        This is a bigram tag model (see fit()'s transition counting),
        so only the single most recent tag in tag_context matters; an
        empty tag_context (sentence start, or a context of words this
        tagger couldn't tag) falls back to the initial-tag distribution.
        """
        if not tag_context:
            return {tag: math.exp(self._log_initial(tag)) for tag in self.tags}
        prev_tag = tag_context[-1]
        return {tag: math.exp(self._log_transition(prev_tag, tag)) for tag in self.tags}
