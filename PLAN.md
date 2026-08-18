# N-Ticipate — Implementation Plan

An NLP text auto-completion engine (bigrams/trigrams/n-grams + HMM POS
tagging), packaged as a desktop app that lives in the system tray and
works across any application on the machine — similar in spirit to
Grammarly, but built from scratch as a college NLP lab project.

Two prediction modes share one scoring engine:

- **Word completion** — `rec` → `recommend` / `receive` / `recent`
- **Next-word prediction** — `I would like to ` → `know` / `thank` / `see`

The n-gram engine generates candidates; an HMM POS tagger reranks them
by grammatical plausibility. That reranking is a real design decision,
not just extra coursework bolted on: if the context is `the quick
brown ___`, a tag-bigram model knows the next tag is very likely `NOUN`
or `ADJ`, so verb candidates get demoted. It also produces a genuine
ablation study for the report — accuracy with vs. without POS
reranking (Phase 6).

## Runtime pipeline

```
keystroke (any app, anywhere)
        |
        v
global keystroke hook  ---------------------------  Phase 7
        |
        v
context buffer (last two words + current prefix)
        |
        v
n-gram candidate generator (trie prefix + backoff) -  Phases 2-3
        |
        v
HMM POS reranker (boosts grammatically likely tags) -  Phase 6
        |
        v
tray overlay (Tab accepts, Esc dismisses)  ---------  Phase 7
```

Every stage has to run in well under the per-keystroke debounce window
(~30–50ms) — that constraint shapes most of the design choices below
(stupid-backoff over Kneser-Ney in the shipped model, pruning before
packaging, etc.).

---

## Phase 0 — Scaffolding

Repo structure, `config.yaml` (every tunable parameter for every later
phase in one place — smoothing method, HMM language, reranking alpha,
hotkeys), a cached config loader, `setup_env.py` for one-time corpus
downloads, and a pytest smoke-test suite that passes before any real
code exists.

**Status: done.**

## Phase 1 — Preprocessing

`clean_text` → `segment_sentences` → `tokenize` → `build_vocab` /
`apply_unk` → `build_truecase_map` / `apply_truecase` → `pad_sentence`
→ `train_dev_test_split`.

Deliberately keeps stopwords and punctuation (removing them is right
for classification, wrong for language modeling — "of the" is exactly
what a bigram model needs to predict). Counting is case-insensitive
(`lowercase_for_counts` in config) with a truecase map to restore
natural casing at suggestion time, so the app doesn't suggest "india"
mid-sentence. NLTK's Punkt/TreebankWordTokenizer are the primary
backend, with a regex fallback so the module works before corpora are
downloaded.

**Deliverables:** before/after token counts, Zipf plot, type-token
ratio, vocab coverage curve. → `notebooks/01_preprocessing.ipynb`

**Status: done.**

## Phase 2 — N-gram language model

Unigram/bigram/trigram counting, with four smoothing methods:

| Method                  | Why it's implemented                                                   |
| ----------------------- | ---------------------------------------------------------------------- |
| MLE                     | Baseline — demonstrates the zero-probability problem on unseen n-grams |
| Laplace / add-k         | Classic textbook smoothing                                             |
| Stupid backoff          | Fast, no renormalization — what the running app actually uses          |
| Interpolated Kneser-Ney | The quality ceiling for comparison                                     |

Evaluated by perplexity on held-out data, swept across n and smoothing
method. Includes pruning (drop rare n-grams, cap continuations per
context) for app packaging, with an explicit before/after size vs.
perplexity trade-off measurement.

**Deliverables:** perplexity table, sample generated sentences per
model order, model size comparison. → `notebooks/02_ngram_models.ipynb`

**Status: done.**

## Phase 3 — Prediction engine

A prefix trie for word completion, plus a unified `Predictor.predict()`
that handles both prediction modes and blends in a personalization
layer: a small per-user n-gram model trained continuously on what the
user actually types, interpolated with the base model
(`λ · P_user + (1−λ) · P_base`, λ growing as more user data accumulates).
Deliberately skips `<UNK>`-ing the user's own vocabulary — that's the
whole point, capturing names and jargon the base corpus would discard.

**Deliverables:** latency benchmark (p95 target ~20–50ms), keystroke
savings ratio, hit@1 / hit@3 / hit@5 on held-out text.
→ `notebooks/03_prediction_engine.ipynb`

**Status: done.**

## Phase 4 — HMM POS tagger, English

Implemented from scratch — transition matrix, emission matrix, initial
distribution, all Laplace-smoothed — with log-space Viterbi decoding.
`nltk.HiddenMarkovModelTrainer` is used only as an optional correctness
cross-check, never in the shipped path. Unknown words fall back to
suffix heuristics (`-ing`/`-ed` → VERB, `-ly` → ADV, `-tion` → NOUN,
capitalized → proper-noun tag).

**Deliverables:** accuracy on held-out set, confusion matrix,
comparison against a most-frequent-tag baseline and (optionally)
NLTK's own tagger, a worked-by-hand Viterbi trellis for one sentence.
→ `notebooks/04_hmm_english.ipynb`

**Status: done.**

## Phase 5 — HMM tagger, regional language (Hindi)

The exact same `HMMTagger` class, zero code changes — only the corpus
differs, plus the (already-built) Devanagari branch of the suffix
heuristic. Devanagari text normalized via Unicode NFC (and
`indic-nlp-library` where available); danda (।/॥) handled as sentence
punctuation, not merged into the preceding word.

**Deliverables:** side-by-side accuracy table (English vs. Hindi), TTR
and OOV-rate comparison, error analysis on the most-confused tag pairs.
→ `notebooks/05_hmm_regional.ipynb`

**Status: done.**

## Phase 6 — POS-aware reranking

Wires Phases 4/5 into the Phase 3 predictor. The context is tagged via
the HMM's own `viterbi()`; each candidate word gets a context-free
"typical tag" guess; final ranking score is
`log P(word|context) + alpha · log P(tag(word) | tag_context)`.

**Deliverables:** hit@k with vs. without reranking (the core ablation),
reported honestly even when the effect is small or mixed at sample
scale — a near-zero delta on a small corpus is a legitimate finding,
not a failed experiment.

**Status: done.**

## Phase 7 — Desktop app

The riskiest phase, done last, after the NLP core is provably working.

- **Tray icon** — `pystray` + Pillow (or `QSystemTrayIcon`/PySide6 for
  a richer menu): enable/disable, language toggle, settings, "retrain
  on my typing," quit.
- **Global keystroke capture** — `pynput.keyboard.Listener`, a rolling
  context buffer that resets on Enter/click/focus-change, debounced at
  ~50ms.
- **Caret positioning** — the hard part. `GetGUIThreadInfo` via
  `ctypes` works for most Win32 apps; Electron apps and browsers often
  won't cooperate. Fallback: anchor near the mouse cursor.
- **Overlay window** — frameless, always-on-top, `WS_EX_NOACTIVATE` so
  it never steals focus (the bug that makes these apps unusable).
- **Injection** — `pynput` types the remaining characters on accept;
  clipboard+paste is faster for long completions but must save/restore
  the user's existing clipboard contents.

**Known risks:** antivirus/Defender may flag a global keyboard hook as
a keylogger (develop with an exclusion folder, mention it in the demo);
some apps need admin rights. Hard requirement: everything stays local,
nothing leaves the machine, the buffer holds only the current sentence,
and a password-field heuristic disables capture entirely.

**Fallback if the global hook proves unworkable:** a built-in
Tkinter/Qt editor window with live suggestions. Less impressive as a
product, but demonstrates every NLP component correctly, and no lab
marks are lost.

**Status: not started.**

## Phase 8 — Evaluation, packaging, report

Bundle with PyInstaller into a single `.exe` with the pruned model
embedded. Final metrics table:

| Metric                        | What it shows                               |
| ----------------------------- | ------------------------------------------- |
| Perplexity                    | Language model quality, per order/smoothing |
| Keystroke savings ratio       | The headline product number                 |
| hit@1 / @3 / @5               | Prediction accuracy                         |
| p50 / p95 latency             | Real-time usability                         |
| Tagger accuracy               | English vs. regional, per tag               |
| Model size, RAM, startup time | Practicality                                |

**Status: not started.**

---

## Suggested sequencing

Phases 1→2→3 first, so there's a working predictor with real numbers
early. Then 4→5→6 for the POS-tagging coursework. Phase 7 last, since
it's the one most likely to blow its time estimate — and if the global
hook doesn't pan out, the Tkinter/Qt fallback editor still delivers
every required NLP component cleanly.
