"""
Phase 5 tests: HMMTagger against the real hand-tagged Hindi corpus
(data/raw/sample_tagged_hindi.conll), not synthetic data -- guards
against corpus-file corruption/format regressions, and confirms the
Phase 4 HMMTagger class works unmodified on Devanagari input, which is
the whole point of "same class serves both experiments."

Run with: pytest tests/test_hmm_regional.py -v
"""

import random

from nticipate.hmm import HMMTagger
from nticipate.config import resolve_path


def _load_conll(path):
    sentences, current = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if current:
                    sentences.append(current)
                    current = []
                continue
            word, tag = line.split("\t")
            current.append((word, tag))
        if current:
            sentences.append(current)
    return sentences


def _load_hindi_split():
    sentences = _load_conll(resolve_path("data/raw/sample_tagged_hindi.conll"))
    rng = random.Random(42)
    shuffled = list(sentences)
    rng.shuffle(shuffled)
    split = int(len(shuffled) * 0.8)
    return shuffled[:split], shuffled[split:]


def test_hindi_corpus_file_loads_and_is_nonempty():
    sentences = _load_conll(resolve_path("data/raw/sample_tagged_hindi.conll"))
    assert len(sentences) > 0
    for sentence in sentences:
        assert len(sentence) > 0
        for word, tag in sentence:
            assert isinstance(word, str) and word
            assert isinstance(tag, str) and tag


def test_hindi_corpus_uses_same_tagset_family_as_english():
    # not identical (small-corpus shuffles can drop a rare tag from a
    # split), but every tag used should be a plausible Universal tag
    universal_tags = {"NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP", "NUM", "CONJ", "PRT", "."}
    sentences = _load_conll(resolve_path("data/raw/sample_tagged_hindi.conll"))
    seen_tags = {tag for sentence in sentences for _, tag in sentence}
    assert seen_tags <= universal_tags


def test_same_hmmtagger_class_fits_hindi_data():
    train, _ = _load_hindi_split()
    tagger = HMMTagger().fit(train)
    assert len(tagger.tags) > 0
    assert len(tagger.vocab) > 0


def test_hindi_tagger_beats_most_common_tag_baseline():
    train, test = _load_hindi_split()
    tagger = HMMTagger().fit(train)

    acc = tagger.accuracy(test)
    most_common = tagger._most_common_tag()
    total = sum(len(s) for s in test)
    baseline_correct = sum(1 for s in test for _, t in s if t == most_common)
    baseline_acc = baseline_correct / total if total else 0.0

    assert acc > baseline_acc


def test_devanagari_suffix_heuristic_used_for_oov_hindi_words():
    train, _ = _load_hindi_split()
    tagger = HMMTagger().fit(train)

    # a word not in this split's vocab, with an unambiguous verb suffix
    oov_word = "पढ़ना"  # "to read" -- infinitive marker ना
    if oov_word not in tagger.vocab:
        assert tagger._guess_tag_by_suffix(oov_word) == "VERB"


def test_viterbi_runs_on_devanagari_tokens():
    train, test = _load_hindi_split()
    tagger = HMMTagger().fit(train)
    tokens = [word for word, _ in test[0]]
    tags = tagger.viterbi(tokens)
    assert len(tags) == len(tokens)
