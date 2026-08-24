"""Rebuild every artefact the app loads, from the corpora in data/raw.

Runs Phases 1, 2, 4 and 5 end to end and writes:

    data/processed/modern.json          Corpus  (splits, vocab, truecase map)
    data/models/truecase.json           truecase map alone (what the app loads)
    data/models/ngram_trigram_pruned.pkl
    data/models/hmm_english.pkl
    data/models/hmm_hindi.pkl

Reads its paths from config.yaml, so the config is the single place that
records which corpus the shipped models came from. No module is modified.

    python scripts/retrain.py                 # everything
    python scripts/retrain.py --only lm       # lm | english | hindi
    python scripts/retrain.py --limit 20000   # smoke test on 20k sentences
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nticipate import hmm as hmm_mod
from nticipate import ngram as ngram_mod
from nticipate import preprocess as pre
from nticipate.config import get, resolve_path

ROOT = Path(__file__).resolve().parent.parent


def _t(started: float) -> str:
    return f"{time.perf_counter() - started:.1f}s"


def build_lm(limit: int | None) -> None:
    corpus_path = get("app.models.corpus", "data/processed/modern.json")
    model_path = get("app.models.ngram", "data/models/ngram_trigram_pruned.pkl")
    source = get("ngram.corpus.english")

    print(f"[1] preprocessing  {source}")
    started = time.perf_counter()
    sentences = pre.load_corpus_sentences(source, limit=limit)
    print(f"    {len(sentences):,} sentences loaded ({_t(started)})")

    started = time.perf_counter()
    corpus = pre.preprocess_sentences(sentences)
    stats = pre.corpus_stats(corpus.train)
    print(f"    train {len(corpus.train):,} / dev {len(corpus.dev):,} "
          f"/ test {len(corpus.test):,} sentences ({_t(started)})")
    print(f"    tokens {stats['tokens']:,}  types {stats['types']:,}  "
          f"TTR {stats['type_token_ratio']:.4f}")
    print(f"    OOV on dev {pre.oov_rate(corpus.dev, corpus.vocab) * 100:.2f}%")
    saved = corpus.save(corpus_path)
    print(f"    -> {saved}  ({saved.stat().st_size / 1e6:.1f} MB)")
    truecase_path = get("app.models.truecase", "data/models/truecase.json")
    saved = corpus.save_truecase(truecase_path)
    print(f"    -> {saved}  ({saved.stat().st_size / 1e6:.1f} MB)")

    print(f"[2] n-gram  order={get('ngram.max_order')} "
          f"smoothing={get('ngram.smoothing')}")
    started = time.perf_counter()
    model = ngram_mod.train_model(
        corpus.train, order=get("ngram.max_order", 3),
        smoothing=get("ngram.smoothing", "stupid_backoff"),
    )
    print(f"    fitted {model.stats().total_ngrams:,} n-grams ({_t(started)})")

    if get("ngram.pruning.enabled", True):
        before = model.stats().total_ngrams
        model.prune(
            min_count=get("ngram.pruning.min_count", 2),
            max_continuations=get("ngram.pruning.max_continuations", 50),
        )
        after = model.stats().total_ngrams
        print(f"    pruned {before:,} -> {after:,} n-grams "
              f"({100 * (1 - after / before):.1f}% dropped)")

    ppl = model.perplexity(corpus.dev)
    print(f"    perplexity (dev, pruned) {ppl:,.1f}")
    saved = model.save(model_path)
    print(f"    -> {saved}  ({saved.stat().st_size / 1e6:.1f} MB)")


def build_tagger(language: str) -> None:
    out = get(f"app.models.tagger" if language == "english"
              else "app.models.tagger_hindi")
    print(f"[{'4' if language == 'english' else '5'}] HMM tagger  {language}")
    started = time.perf_counter()
    sentences = hmm_mod.load_tagged_sentences(language)
    train, test = hmm_mod.train_test_split_tagged(sentences)
    tagger = hmm_mod.HMMTagger().fit(train)
    result = tagger.evaluate(test)
    baseline = hmm_mod.MostFrequentTagBaseline().fit(train).evaluate(test)
    print(f"    {len(sentences):,} sentences, {len(tagger.tags)} tags, "
          f"vocab {tagger.vocab_size:,} ({_t(started)})")
    print(f"    accuracy {result.accuracy * 100:.2f}%  "
          f"(known {result.known_accuracy * 100:.2f}%, "
          f"unknown {result.unknown_accuracy * 100:.2f}%, "
          f"OOV {result.oov_rate * 100:.2f}%)")
    print(f"    most-frequent-tag baseline {baseline.accuracy * 100:.2f}%")
    saved = tagger.save(out)
    print(f"    -> {saved}  ({saved.stat().st_size / 1e6:.1f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=["lm", "english", "hindi"], action="append")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap sentences loaded for the language model")
    args = ap.parse_args()
    wanted = set(args.only or ["lm", "english", "hindi"])

    if "lm" in wanted:
        build_lm(args.limit)
    for language in ("english", "hindi"):
        if language in wanted:
            build_tagger(language)

    print("\nDone. Sanity check:")
    print("  .venv/Scripts/python.exe -m nticipate.app --check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
