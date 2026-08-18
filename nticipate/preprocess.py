"""
Phase 1: preprocessing.

Pipeline (see notebooks/01_preprocessing.ipynb for the lab writeup):
    raw text -> clean_text -> segment_sentences -> tokenize
             -> build_vocab / apply_unk -> pad_sentence -> train_dev_test_split

Each function is independently testable and independently demoable --
that's why this stays as separate small functions rather than one big
pipeline() blob. See tests/test_preprocess.py.

NLTK is the primary tokenizer/segmenter backend (Punkt for sentences,
TreebankWordTokenizer for words) since it correctly handles English
contraction-splitting ("don't" -> "do" + "n't"). Both functions fall
back to a regex-based implementation if NLTK or its data isn't
available yet, so the rest of the module (and its tests) don't hard-
depend on `python setup_env.py` having been run first. The regex
fallback is a simplified approximation -- it does NOT split
contractions -- run setup_env.py for the real behaviour.
"""

import re
import random
import unicodedata
from collections import Counter

from nticipate.config import load_config

CFG = load_config()

# ---------------------------------------------------------------------------
# Optional NLTK backend
# ---------------------------------------------------------------------------
_NLTK_AVAILABLE = True
try:
    from nltk.tokenize import sent_tokenize as _nltk_sent_tokenize
    from nltk.tokenize import TreebankWordTokenizer as _TreebankWordTokenizer

    _treebank = _TreebankWordTokenizer()
except ImportError:
    _NLTK_AVAILABLE = False


# ---------------------------------------------------------------------------
# clean_text
# ---------------------------------------------------------------------------
_SMART_CHAR_MAP = {
    "\u2018": "'", "\u2019": "'",   # single smart quotes
    "\u201c": '"', "\u201d": '"',  # double smart quotes
    "\u2013": "-", "\u2014": "-",  # en/em dash
    "\u00a0": " ",                 # non-breaking space
}
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Strip markup, normalize unicode, collapse whitespace."""
    if not text:
        return ""
    # NFC: compose accented / Devanagari combining characters consistently
    text = unicodedata.normalize("NFC", text)
    for src, dst in _SMART_CHAR_MAP.items():
        text = text.replace(src, dst)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _CONTROL_CHARS_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# segment_sentences
# ---------------------------------------------------------------------------
_DEVANAGARI_DANDA_RE = re.compile(r"[।॥]")
_DANDA_SPLIT_RE = re.compile(r"(?<=[।॥])\s*")
_ENGLISH_FALLBACK_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\'(])')


def segment_sentences(text: str) -> list[str]:
    """Split into sentences. Punkt for English; danda-aware for Devanagari."""
    if not text:
        return []

    # Devanagari text uses danda (।) / double danda (॥) as sentence-final
    # punctuation instead of periods -- detect and split on that first,
    # regardless of NLTK availability, since Punkt doesn't know danda.
    if _DEVANAGARI_DANDA_RE.search(text):
        parts = _DANDA_SPLIT_RE.split(text)
        return [p.strip() for p in parts if p.strip()]

    if _NLTK_AVAILABLE:
        try:
            return [s.strip() for s in _nltk_sent_tokenize(text) if s.strip()]
        except LookupError:
            pass  # punkt data not downloaded -- fall through to regex

    sentences = _ENGLISH_FALLBACK_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


# ---------------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------------
# Devanagari Unicode block is U+0900-U+097F, but danda/double-danda
# (U+0964 '।', U+0965 '॥') live inside that same block and are
# punctuation, not word characters -- excluded from the word-block
# range below so "है।" tokenizes as "है" + "।", not one glued token.
_DANDA_TOKEN_RE = re.compile(r"[।॥]|[^।॥]+")

_FALLBACK_TOKEN_RE = re.compile(
    r"[\u0900-\u0963\u0966-\u097F]+"  # Devanagari word block, minus danda
    r"|[।॥]"                            # danda punctuation, its own token
    r"|[A-Za-z]+(?:'[A-Za-z]+)?"         # English word, optionally with an apostrophe
    r"|\d+(?:[.,]\d+)*"                  # numbers
    r"|[^\sA-Za-z0-9\u0900-\u097F]"       # any other single punctuation character
)


def tokenize(sentence: str) -> list[str]:
    """Word-tokenize a single sentence. Keep contractions split (do / n't).

    Contraction-splitting is only guaranteed on the NLTK path
    (TreebankWordTokenizer). The regex fallback keeps contractions as a
    single token (e.g. "don't") -- see module docstring.
    """
    if not sentence:
        return []

    if _NLTK_AVAILABLE:
        try:
            tokens = _treebank.tokenize(sentence)
        except LookupError:
            pass
        else:
            # Treebank doesn't know danda/double-danda, so it leaves "है।" as
            # one token -- split them back out to match the regex fallback.
            return [
                piece
                for token in tokens
                for piece in _DANDA_TOKEN_RE.findall(token)
                if piece.strip()
            ]

    return [t for t in _FALLBACK_TOKEN_RE.findall(sentence) if t.strip()]


# ---------------------------------------------------------------------------
# vocabulary
# ---------------------------------------------------------------------------
UNK_TOKEN = "<UNK>"
START_TOKEN = "<s>"
END_TOKEN = "</s>"


def _counting_key(token: str) -> str:
    """The form a token is counted/looked-up under. Controlled by
    CFG['preprocessing']['lowercase_for_counts'] -- counting collapses
    case ("The" and "the" are the same n-gram context), but the surface
    form typed by the user is preserved separately via the truecase map
    below and restored at suggestion time.
    """
    return token.lower() if CFG["preprocessing"]["lowercase_for_counts"] else token


def build_vocab(tokenized_sentences: list[list[str]], min_freq: int | None = None) -> set[str]:
    """Return the vocabulary (of counting-keys, see _counting_key), collapsing
    rare words into <UNK>. min_freq defaults to CFG['preprocessing']['min_token_freq'].
    """
    min_freq = min_freq if min_freq is not None else CFG["preprocessing"]["min_token_freq"]
    counts = Counter()
    for sentence in tokenized_sentences:
        counts.update(_counting_key(t) for t in sentence)

    vocab = {word for word, count in counts.items() if count >= min_freq}
    vocab.update({UNK_TOKEN, START_TOKEN, END_TOKEN})
    return vocab


def apply_unk(tokens: list[str], vocab: set[str]) -> list[str]:
    """Map each token to its counting-key and replace any key not in
    vocab with '<UNK>'. Output is in counting-key form (lowercased, if
    CFG['preprocessing']['lowercase_for_counts']) -- use build_truecase_map
    + apply_truecase to restore natural casing for display.
    """
    result = []
    for t in tokens:
        key = _counting_key(t)
        result.append(key if key in vocab else UNK_TOKEN)
    return result


def build_truecase_map(tokenized_sentences: list[list[str]]) -> dict[str, str]:
    """Map counting-key (usually lowercased) -> most frequent original-case
    spelling seen in the corpus, e.g. 'india' -> 'India'.

    Counting happens on lowercased tokens so 'The' and 'the' aren't split
    into separate vocab/n-gram entries -- but a suggestion shown to the
    user should use natural casing, not always-lowercase. This map is
    how Phase 3's predictor restores that at suggestion time.
    """
    casing_counts: dict[str, Counter] = {}
    for sentence in tokenized_sentences:
        for token in sentence:
            key = _counting_key(token)
            casing_counts.setdefault(key, Counter())[token] += 1
    return {key: counter.most_common(1)[0][0] for key, counter in casing_counts.items()}


def apply_truecase(tokens: list[str], truecase_map: dict[str, str]) -> list[str]:
    """Restore natural casing to counting-key tokens using a truecase map.
    Falls back to the token unchanged if it isn't in the map (e.g. <UNK>).
    """
    return [truecase_map.get(t, t) for t in tokens]


def pad_sentence(tokens: list[str], n: int) -> list[str]:
    """Add (n-1) <s> markers at the start and one </s> at the end."""
    if n < 1:
        raise ValueError("n must be >= 1")
    return [START_TOKEN] * (n - 1) + list(tokens) + [END_TOKEN]


def train_dev_test_split(
    sentences: list[list[str]],
) -> tuple[list[list[str]], list[list[str]], list[list[str]]]:
    """Split using CFG['preprocessing'] ratios and CFG['project']['seed']."""
    pp = CFG["preprocessing"]
    train_r, dev_r, test_r = pp["train_split"], pp["dev_split"], pp["test_split"]
    if abs((train_r + dev_r + test_r) - 1.0) > 1e-6:
        raise ValueError("train/dev/test splits in config.yaml must sum to 1.0")

    rng = random.Random(CFG["project"]["seed"])
    shuffled = list(sentences)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_r)
    n_dev = int(n * dev_r)

    train = shuffled[:n_train]
    dev = shuffled[n_train : n_train + n_dev]
    test = shuffled[n_train + n_dev :]
    return train, dev, test


# ---------------------------------------------------------------------------
# convenience: full pipeline for a batch of raw documents
# ---------------------------------------------------------------------------
def preprocess_corpus(
    raw_documents: list[str], min_freq: int | None = None
) -> tuple[list[list[str]], set[str], dict[str, str]]:
    """
    Run clean_text -> segment_sentences -> tokenize over a list of raw
    documents (each a big string), then build a shared vocabulary,
    truecase map, and apply <UNK>. Returns
    (unk_applied_sentences, vocab, truecase_map) -- sentences are still
    un-padded; padding is order-specific and happens later, in ngram.py,
    once you know n.
    """
    tokenized_sentences: list[list[str]] = []
    for doc in raw_documents:
        cleaned = clean_text(doc)
        for sentence in segment_sentences(cleaned):
            tokens = tokenize(sentence)
            if tokens:
                tokenized_sentences.append(tokens)

    vocab = build_vocab(tokenized_sentences, min_freq=min_freq)
    truecase_map = build_truecase_map(tokenized_sentences)
    unk_applied = [apply_unk(sent, vocab) for sent in tokenized_sentences]
    return unk_applied, vocab, truecase_map
