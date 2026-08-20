"""Phase 1 — text preprocessing for language modelling.

Pipeline::

    clean_text -> segment_sentences -> tokenize -> build_vocab / apply_unk
               -> build_truecase_map / apply_truecase -> pad_sentence
               -> train_dev_test_split

Two decisions drive the whole module:

**Stopwords and punctuation stay.** Stripping them is correct for text
classification and wrong for language modelling — "of the" is exactly the
bigram an autocomplete engine has to predict, and a sentence-final "." is a
real event a trigram model should learn.

**Counting is case-insensitive, output is truecased.** ``the`` and ``The``
are the same word for statistics, so they are folded together before counting.
A separate truecase map records the casing each word usually appears in, so
the running app suggests ``India`` and not ``india`` — without splitting the
counts across two surface forms.

The NLTK Punkt splitter and Treebank tokenizer are the primary backends; a
regex fallback keeps the module usable before ``setup_env.py`` has run, and
handles Devanagari (including the danda ``।``) for Phase 5.
"""

from __future__ import annotations

import json
import random
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from nticipate.config import data_dir, get, resolve_path

Token = str
Sentence = list[Token]

# --------------------------------------------------------------------------
# Patterns
# --------------------------------------------------------------------------

URL_RE = re.compile(r"\b(?:https?://|www\.)\S+\b|\b\S+\.(?:com|org|net|io|edu)\S*",
                    re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
WHITESPACE_RE = re.compile(r"\s+")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

#: Sentence-ending punctuation, including the Devanagari danda and double danda.
SENTENCE_END = ".!?।॥"
SENTENCE_SPLIT_RE = re.compile(rf"(?<=[{re.escape(SENTENCE_END)}])\s+")

#: Combining marks that are part of a word, not separate tokens. Needed
#: because a Devanagari vowel sign (matra) such as U+093E is Unicode category
#: Mc, not a letter, so ``\w`` and ``[^\W\d_]`` both reject it — and "भारत"
#: would tokenise as ['भ', 'ा', 'रत']. Covers the Devanagari signs, matras,
#: virama and nukta (Phase 5), plus the generic combining diacriticals.
_MARK = r"̀-ͯऀ-ःऺ-ॏ॑-ॗॢ-ॣ"

#: A word character: any letter, or any of the marks above. The danda (U+0964)
#: falls outside these ranges on purpose — it is punctuation, not a word part.
_WCHAR = rf"(?:[^\W\d_]|[{_MARK}])"
_WORD = rf"{_WCHAR}+(?:['’-]{_WCHAR}+)*"
_NUMBER = r"\d+(?:[.,]\d+)*"
_PUNCT = r"[^\w\s]"
TOKEN_RE = re.compile(rf"{_WORD}|{_NUMBER}|{_PUNCT}", re.UNICODE)

#: Treebank-style contraction splits, applied by the regex fallback so its
#: output stays comparable with NLTK's tokenizer.
CONTRACTIONS = [
    (re.compile(r"\b(can)(not)\b", re.I), r"\1 \2"),
    (re.compile(r"(\w)(n['’]t)\b", re.I), r"\1 \2"),
    (re.compile(r"(\w)(['’](?:s|re|ve|ll|d|m))\b", re.I), r"\1 \2"),
]


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------

def clean_text(
    text: str,
    normalize: str | None = None,
    strip_urls: bool | None = None,
    strip_emails: bool | None = None,
) -> str:
    """Normalise unicode, drop URLs/emails and control chars, squeeze whitespace.

    ``normalize`` defaults to ``preprocessing.normalize_unicode`` (NFC), which
    matters well beyond tidiness for Phase 5: Devanagari can encode the same
    grapheme either precomposed or as a base plus a combining mark, and the two
    forms would otherwise count as different words.
    """
    if normalize is None:
        normalize = get("preprocessing.normalize_unicode", "NFC")
    if strip_urls is None:
        strip_urls = get("preprocessing.strip_urls", True)
    if strip_emails is None:
        strip_emails = get("preprocessing.strip_emails", True)

    if normalize:
        text = unicodedata.normalize(normalize, text)
    text = CONTROL_RE.sub(" ", text)
    if strip_emails:
        text = EMAIL_RE.sub(" ", text)
    if strip_urls:
        text = URL_RE.sub(" ", text)
    # Curly quotes and dashes fold to ASCII so "don't" and "don’t" are one word.
    text = text.translate(str.maketrans({
        "‘": "'", "’": "'", "“": '"', "”": '"',
        "–": "-", "—": "-", " ": " ",
    }))
    return WHITESPACE_RE.sub(" ", text).strip()


# --------------------------------------------------------------------------
# Sentence segmentation
# --------------------------------------------------------------------------

def _punkt_available() -> bool:
    try:
        import nltk

        nltk.data.find("tokenizers/punkt_tab")
        return True
    except Exception:
        try:
            import nltk

            nltk.data.find("tokenizers/punkt")
            return True
        except Exception:
            return False


def segment_sentences(text: str, splitter: str | None = None) -> list[str]:
    """Split text into sentences.

    Uses NLTK Punkt when available (it knows that "Dr." is not a sentence end),
    otherwise a regex that splits after ``.!?`` and the Devanagari danda.
    Punkt has no Hindi model, so Hindi always takes the regex path — which is
    fine, because the danda is an unambiguous terminator with no abbreviation
    use, unlike the English full stop.
    """
    if splitter is None:
        splitter = get("preprocessing.sentence_splitter", "punkt")
    text = text.strip()
    if not text:
        return []

    if splitter == "punkt" and not contains_devanagari(text) and _punkt_available():
        try:
            import nltk

            return [s for s in (s.strip() for s in nltk.sent_tokenize(text)) if s]
        except Exception:
            pass  # fall through to regex

    return [s for s in (s.strip() for s in SENTENCE_SPLIT_RE.split(text)) if s]


def contains_devanagari(text: str) -> bool:
    """True when any character is Devanagari. Phase 5 branches on this."""
    return any("ऀ" <= ch <= "ॿ" for ch in text)


# --------------------------------------------------------------------------
# Tokenisation
# --------------------------------------------------------------------------

def _treebank_available() -> bool:
    try:
        from nltk.tokenize import TreebankWordTokenizer  # noqa: F401

        return True
    except Exception:
        return False


def tokenize(sentence: str, tokenizer: str | None = None) -> Sentence:
    """Tokenise one sentence, keeping punctuation as its own tokens.

    Punctuation is kept deliberately (see the module docstring). The regex
    fallback mimics the Treebank contraction splits ("don't" -> "do n't") so
    the two backends produce comparable vocabularies.
    """
    if tokenizer is None:
        tokenizer = get("preprocessing.tokenizer", "nltk")
    sentence = sentence.strip()
    if not sentence:
        return []

    if tokenizer == "nltk" and not contains_devanagari(sentence) and _treebank_available():
        try:
            from nltk.tokenize import TreebankWordTokenizer

            return TreebankWordTokenizer().tokenize(sentence)
        except Exception:
            pass

    text = sentence
    for pattern, repl in CONTRACTIONS:
        text = pattern.sub(repl, text)
    return TOKEN_RE.findall(text)


def tokenize_text(text: str, **kwargs) -> list[Sentence]:
    """Clean, segment and tokenise a raw document into a list of sentences."""
    cleaned = clean_text(text)
    return [
        toks for toks in (tokenize(s, **kwargs) for s in segment_sentences(cleaned))
        if toks
    ]


# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------

@dataclass
class Vocab:
    """Closed vocabulary plus the ``<UNK>`` / boundary symbols."""

    tokens: set[str] = field(default_factory=set)
    counts: Counter = field(default_factory=Counter)
    unk: str = "<UNK>"
    bos: str = "<s>"
    eos: str = "</s>"

    def __contains__(self, token: str) -> bool:
        return token in self.tokens

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def specials(self) -> tuple[str, str, str]:
        return (self.unk, self.bos, self.eos)

    def to_dict(self) -> dict:
        return {
            "tokens": sorted(self.tokens),
            "counts": dict(self.counts),
            "unk": self.unk,
            "bos": self.bos,
            "eos": self.eos,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Vocab":
        return cls(
            tokens=set(data["tokens"]),
            counts=Counter(data.get("counts", {})),
            unk=data.get("unk", "<UNK>"),
            bos=data.get("bos", "<s>"),
            eos=data.get("eos", "</s>"),
        )


def build_vocab(
    sentences: Iterable[Sentence],
    min_freq: int | None = None,
    max_size: int | None = None,
    lowercase: bool | None = None,
) -> Vocab:
    """Count tokens and keep those passing the frequency / size cutoffs.

    Everything below the cutoff is mapped to ``<UNK>`` by :func:`apply_unk`.
    That is not merely a size optimisation: without an ``<UNK>`` class, a model
    assigns probability zero to any unseen word and perplexity on held-out text
    becomes infinite (demonstrated in Phase 2).
    """
    if min_freq is None:
        min_freq = get("preprocessing.min_token_freq", 2)
    if max_size is None:
        max_size = get("preprocessing.max_vocab_size", 0)
    if lowercase is None:
        lowercase = get("preprocessing.lowercase_for_counts", True)

    counts: Counter = Counter()
    for sentence in sentences:
        counts.update(t.lower() if lowercase else t for t in sentence)

    kept = {t for t, c in counts.items() if c >= min_freq}
    if max_size:
        # most_common is stable on ties, so the vocabulary is reproducible.
        # The membership set is hoisted out of the comprehension on purpose:
        # rebuilding it per candidate turns this into an O(types * vocab) loop
        # that costs minutes on a corpus the size of Brown.
        ranked = [t for t, _ in counts.most_common() if t in kept]
        kept = set(ranked[:max_size])

    vocab = Vocab(
        tokens=kept,
        counts=counts,
        unk=get("preprocessing.unk_token", "<UNK>"),
        bos=get("preprocessing.bos_token", "<s>"),
        eos=get("preprocessing.eos_token", "</s>"),
    )
    return vocab


def apply_unk(sentence: Sentence, vocab: Vocab, lowercase: bool | None = None) -> Sentence:
    """Replace out-of-vocabulary tokens with ``<UNK>``."""
    if lowercase is None:
        lowercase = get("preprocessing.lowercase_for_counts", True)
    out = []
    for token in sentence:
        key = token.lower() if lowercase else token
        out.append(key if key in vocab.tokens else vocab.unk)
    return out


# --------------------------------------------------------------------------
# Truecasing
# --------------------------------------------------------------------------

def build_truecase_map(sentences: Iterable[Sentence]) -> dict[str, str]:
    """Map each lowercased word to the surface form it usually appears in.

    Sentence-initial tokens are skipped when counting: their capital is a
    typographic convention, not evidence about the word. Counting them would
    make every sentence-initial ``The`` argue that "the" is normally
    capitalised. Words that only ever appear sentence-initially therefore get
    no entry, and :func:`apply_truecase` leaves them lowercase — the honest
    answer when the corpus carries no evidence either way.
    """
    surface: dict[str, Counter] = {}
    for sentence in sentences:
        for index, token in enumerate(sentence):
            if index == 0:
                continue
            surface.setdefault(token.lower(), Counter())[token] += 1

    truecase = {}
    for lower, forms in surface.items():
        best, best_count = forms.most_common(1)[0]
        # Only override the lowercase default when the cased form actually
        # dominates; "Apple" the company should not recase every "apple".
        if best != lower and best_count <= forms.get(lower, 0):
            best = lower
        truecase[lower] = best
    return truecase


def apply_truecase(
    sentence: Sentence,
    truecase: dict[str, str],
    capitalize_first: bool = False,
) -> Sentence:
    """Restore natural casing. Set ``capitalize_first`` when rendering a sentence."""
    out = [truecase.get(t.lower(), t.lower()) for t in sentence]
    if capitalize_first and out and out[0][:1].isalpha():
        out[0] = out[0][0].upper() + out[0][1:]
    return out


def truecase_word(word: str, truecase: dict[str, str]) -> str:
    """Truecase a single suggestion — the hot path in the running app."""
    return truecase.get(word.lower(), word)


# --------------------------------------------------------------------------
# Padding
# --------------------------------------------------------------------------

def pad_sentence(
    sentence: Sentence,
    n: int,
    bos: str | None = None,
    eos: str | None = None,
) -> Sentence:
    """Wrap a sentence in ``n-1`` BOS markers and one EOS marker.

    ``n-1`` of them, because a trigram model predicting the first real word
    needs two tokens of left context. The single EOS lets the model learn where
    sentences end, which is what stops generated text running on forever.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if bos is None:
        bos = get("preprocessing.bos_token", "<s>")
    if eos is None:
        eos = get("preprocessing.eos_token", "</s>")
    return [bos] * (n - 1) + list(sentence) + [eos]


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------

def train_dev_test_split(
    sentences: Sequence[Sentence],
    train: float | None = None,
    dev: float | None = None,
    test: float | None = None,
    seed: int | None = None,
) -> tuple[list[Sentence], list[Sentence], list[Sentence]]:
    """Shuffle and split sentences into train / dev / test.

    Shuffled at sentence level with a fixed seed, so runs are reproducible and
    a whole document's vocabulary cannot land entirely in the test set.
    """
    if train is None:
        train = get("preprocessing.split.train", 0.8)
    if dev is None:
        dev = get("preprocessing.split.dev", 0.1)
    if test is None:
        test = get("preprocessing.split.test", 0.1)
    if seed is None:
        seed = get("preprocessing.split.seed", 42)

    total = train + dev + test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split fractions must sum to 1.0, got {total}")

    items = list(sentences)
    random.Random(seed).shuffle(items)

    n = len(items)
    n_train = int(n * train)
    n_dev = int(n * dev)
    return items[:n_train], items[n_train:n_train + n_dev], items[n_train + n_dev:]


# --------------------------------------------------------------------------
# Full pipeline
# --------------------------------------------------------------------------

@dataclass
class Corpus:
    """Everything Phase 2 needs: split sentences, the vocab and the truecase map."""

    train: list[Sentence]
    dev: list[Sentence]
    test: list[Sentence]
    vocab: Vocab
    truecase: dict[str, str]

    @property
    def all_sentences(self) -> list[Sentence]:
        return self.train + self.dev + self.test

    def save(self, path: str | Path) -> Path:
        target = resolve_path(path, create=True)
        payload = {
            "train": self.train,
            "dev": self.dev,
            "test": self.test,
            "vocab": self.vocab.to_dict(),
            "truecase": self.truecase,
        }
        with target.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "Corpus":
        source = resolve_path(path)
        with source.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(
            train=data["train"],
            dev=data["dev"],
            test=data["test"],
            vocab=Vocab.from_dict(data["vocab"]),
            truecase=data["truecase"],
        )


def preprocess_sentences(sentences: Iterable[Sentence], **overrides) -> Corpus:
    """Run the pipeline over already-tokenised sentences.

    The order matters. The truecase map is built from the *tokenised but not
    yet UNK-ed* sentences, because casing evidence for a rare word disappears
    once that word has become ``<UNK>``. The vocabulary is built from the
    training split only — building it from everything would leak test-set
    vocabulary into the model and flatter the Phase 2 perplexity numbers.
    """
    min_sentence_tokens = overrides.pop(
        "min_sentence_tokens", get("preprocessing.min_sentence_tokens", 2)
    )
    lowercase = overrides.pop(
        "lowercase", get("preprocessing.lowercase_for_counts", True)
    )

    sentences = [list(s) for s in sentences if len(s) >= min_sentence_tokens]
    truecase = build_truecase_map(sentences)
    train, dev, test = train_dev_test_split(sentences, **overrides)

    vocab = build_vocab(train, lowercase=lowercase)
    return Corpus(
        train=[apply_unk(s, vocab, lowercase) for s in train],
        dev=[apply_unk(s, vocab, lowercase) for s in dev],
        test=[apply_unk(s, vocab, lowercase) for s in test],
        vocab=vocab,
        truecase=truecase,
    )


def preprocess_corpus(text: str, **overrides) -> Corpus:
    """Run the whole Phase 1 pipeline on a raw document."""
    return preprocess_sentences(tokenize_text(text), **overrides)


# --------------------------------------------------------------------------
# Corpus loading
# --------------------------------------------------------------------------

def load_corpus_sentences(
    source: str | None = None,
    limit: int | None = None,
) -> list[Sentence]:
    """Load training sentences from an NLTK corpus name or a text file.

    NLTK corpora are read through ``.sents()``, never ``.raw()``. The Brown
    corpus's raw form is POS-tagged (``The/at Fulton/np-tl``), so raw text
    would train the language model on tag-suffixed pseudo-words. ``.sents()``
    also gives the corpus's own gold sentence segmentation and tokenisation,
    which beats re-deriving it with Punkt.

    Plain text files go through the full clean/segment/tokenise pipeline.
    Falls back to ``ngram.corpus.fallback_file`` when the NLTK corpus is not
    downloaded, so Phase 1 runs on a fresh clone.
    """
    if source is None:
        source = get("ngram.corpus.english", "brown")

    path = Path(source)
    candidate = path if path.is_absolute() else resolve_path(source)
    if candidate.is_file():
        sentences = tokenize_text(candidate.read_text(encoding="utf-8"))
        return sentences[:limit] if limit else sentences

    try:
        corpus = getattr(__import__("nltk.corpus", fromlist=[source]), source)
        raw_sents = corpus.sents()
        selected = raw_sents[:limit] if limit else raw_sents
        normalize = get("preprocessing.normalize_unicode", "NFC")
        return [
            [unicodedata.normalize(normalize, t) if normalize else t for t in sent]
            for sent in selected
        ]
    except Exception:
        pass

    fallback = get("ngram.corpus.fallback_file")
    if fallback:
        fallback_path = resolve_path(fallback)
        if fallback_path.is_file():
            sentences = tokenize_text(fallback_path.read_text(encoding="utf-8"))
            return sentences[:limit] if limit else sentences

    raise FileNotFoundError(
        f"Could not load corpus {source!r}: not a file, not an available NLTK "
        f"corpus, and no usable fallback. Run: python setup_env.py"
    )


def default_processed_path(name: str = "corpus.json") -> Path:
    return data_dir("processed") / name


# --------------------------------------------------------------------------
# Statistics (Phase 1 deliverables)
# --------------------------------------------------------------------------

def corpus_stats(sentences: Sequence[Sentence]) -> dict:
    """Token/type counts, type-token ratio and hapax rate.

    TTR is reported alongside the raw counts because it is corpus-size
    dependent — comparing the TTR of two corpora of different sizes says more
    about their sizes than their vocabularies (relevant when English and Hindi
    get compared in Phase 5).
    """
    counts: Counter = Counter()
    for sentence in sentences:
        counts.update(sentence)
    tokens = sum(counts.values())
    types = len(counts)
    hapax = sum(1 for c in counts.values() if c == 1)
    return {
        "sentences": len(sentences),
        "tokens": tokens,
        "types": types,
        "type_token_ratio": types / tokens if tokens else 0.0,
        "hapax": hapax,
        "hapax_ratio": hapax / types if types else 0.0,
        "mean_sentence_length": tokens / len(sentences) if sentences else 0.0,
    }


def frequency_table(sentences: Sequence[Sentence], top: int = 20) -> list[tuple[str, int]]:
    counts: Counter = Counter()
    for sentence in sentences:
        counts.update(sentence)
    return counts.most_common(top)


def zipf_points(sentences: Sequence[Sentence]) -> tuple[list[int], list[int]]:
    """Return (ranks, frequencies) for a log-log Zipf plot."""
    counts: Counter = Counter()
    for sentence in sentences:
        counts.update(sentence)
    freqs = sorted(counts.values(), reverse=True)
    return list(range(1, len(freqs) + 1)), freqs


def coverage_curve(
    sentences: Sequence[Sentence],
    steps: Sequence[int] | None = None,
) -> list[tuple[int, float]]:
    """Token coverage achieved by the top-N most frequent types.

    This is the evidence behind ``preprocessing.max_vocab_size``: it shows how
    few types are needed to cover most tokens, and therefore how much of the
    model can be thrown away before the app ships (Phase 2 pruning).
    """
    counts: Counter = Counter()
    for sentence in sentences:
        counts.update(sentence)
    total = sum(counts.values())
    if not total:
        return []

    ordered = [c for _, c in counts.most_common()]
    if steps is None:
        steps = [100, 500, 1000, 2000, 5000, 10000, 20000, 50000]

    curve = []
    for step in steps:
        if step > len(ordered):
            curve.append((len(ordered), 1.0))
            break
        curve.append((step, sum(ordered[:step]) / total))
    return curve


def oov_rate(sentences: Sequence[Sentence], vocab: Vocab) -> float:
    """Fraction of tokens outside the vocabulary — used in the Phase 5 comparison."""
    total = unknown = 0
    for sentence in sentences:
        for token in sentence:
            total += 1
            if token.lower() not in vocab.tokens:
                unknown += 1
    return unknown / total if total else 0.0
