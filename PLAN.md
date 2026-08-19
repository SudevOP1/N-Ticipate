# N-Ticipate — Implementation Plan

An NLP text auto-completion engine (bigrams / trigrams / n-grams + HMM POS
tagging), packaged as a Windows desktop app that lives in the system tray and
suggests text in **any** application — similar in spirit to Grammarly, but
built from scratch as a college NLP lab project.

Two prediction modes share one scoring engine:

| Mode                 | Example                                       |
| -------------------- | --------------------------------------------- |
| Word completion      | `rec` → `recommend` / `receive` / `recent`    |
| Next-word prediction | `I would like to ` → `know` / `thank` / `see` |

The n-gram engine generates candidates; an HMM POS tagger reranks them by
grammatical plausibility. That reranking is a real design decision, not
coursework bolted on: given `the quick brown ___`, a tag-bigram model knows the
next tag is very likely `NOUN` or `ADJ`, so verb candidates get demoted. It
also yields a genuine ablation for the report — hit@k with vs. without POS
reranking (Phase 6).

## Runtime pipeline

```
keystroke (any app, anywhere)
        |
        v
global keystroke hook  -----------------------------  Phase 7
        |
        v
context buffer (last two words + current prefix)
        |
        v
n-gram candidate generator (trie prefix + backoff) --  Phases 2-3
        |
        v
HMM POS reranker (boosts grammatically likely tags) -  Phase 6
        |
        v
tray overlay (Tab accepts, Esc dismisses)  ---------- Phase 7
```

Every stage must run well inside the per-keystroke debounce window (~30–50 ms).
That constraint drives most design choices below (stupid-backoff instead of
Kneser-Ney in the shipped model, pruning before packaging, trie lookup instead
of linear scan).

## Coursework coverage

| Lab requirement                      | Phase   |
| ------------------------------------ | ------- |
| Preprocessing                        | 1       |
| Bigrams, trigrams, n-grams           | 2       |
| POS tagging with HMM (English)       | 4       |
| HMM tagging, regional language       | 5       |
| Application of the above (mini proj) | 3, 6, 7 |
| Evaluation + report                  | 8       |

## Stack

Python 3.12 · NLTK (corpora, Punkt, Treebank tokenizer, optional HMM
cross-check) · NumPy · PyYAML · pystray + Pillow (tray) · pynput (global hook,
injection) · ctypes/Win32 (caret position, overlay window flags) · pytest ·
Jupyter (per-phase deliverable notebooks) · PyInstaller (packaging).

## Repo layout

```
config.yaml               every tunable for every phase, one place
setup_env.py              one-time corpus / model downloads
nticipate/
  config.py               cached YAML loader
  preprocess.py           Phase 1
  ngram.py                Phase 2
  trie.py                 Phase 3
  userprofile.py          Phase 3
  predictor.py            Phases 3 + 6
  hmm.py                  Phases 4 + 5
  app/
    tray.py               Phase 7
    hooks.py              Phase 7
    overlay.py            Phase 7
    injector.py           Phase 7
data/raw|processed|models
notebooks/                01..06, one per phase deliverable
tests/                    pytest, one module per source module
report/notes.md           running log of numbers for the write-up
```

---

## Phase 0 — Scaffolding

Repo structure, `config.yaml`, cached config loader, `setup_env.py`,
`.gitignore`, `requirements.txt`, pytest smoke suite that passes before any
real code exists.

**Done when:** `pytest tests/test_phase0_setup.py` passes and `load_config()`
returns every key later phases read.

## Phase 1 — Preprocessing

`clean_text` → `segment_sentences` → `tokenize` → `build_vocab` / `apply_unk` →
`build_truecase_map` / `apply_truecase` → `pad_sentence` → `train_dev_test_split`.

Deliberately **keeps** stopwords and punctuation — removing them is right for
classification, wrong for language modeling ("of the" is exactly what a bigram
model needs). Counting is case-insensitive (`lowercase_for_counts`) with a
truecase map to restore natural casing at suggestion time, so the app never
suggests "india" mid-sentence. NLTK Punkt / TreebankWordTokenizer primary, with
a regex fallback so the module works before corpora are downloaded.

**Deliverables:** before/after token counts, Zipf plot, type-token ratio, vocab
coverage curve. → `notebooks/01_preprocessing.ipynb`

**Done when:** `tests/test_preprocess.py` green; round-trip
`tokenize → apply_truecase` reproduces original casing on a sample.

## Phase 2 — N-gram language model

Unigram / bigram / trigram counting with four smoothing methods:

| Method                  | Why it's implemented                                                   |
| ----------------------- | ---------------------------------------------------------------------- |
| MLE                     | Baseline — demonstrates the zero-probability problem on unseen n-grams |
| Laplace / add-k         | Classic textbook smoothing                                             |
| Stupid backoff          | Fast, no renormalization — what the running app actually uses          |
| Interpolated Kneser-Ney | Quality ceiling for comparison                                         |

Evaluated by perplexity on held-out data, swept across `n` and smoothing
method. Includes pruning (drop rare n-grams, cap continuations per context) for
packaging, with an explicit size-vs-perplexity trade-off measurement.

**Deliverables:** perplexity table, sample generated sentences per model order,
model size comparison. → `notebooks/02_ngram_models.ipynb`

**Done when:** perplexity is finite for every smoothing method on unseen text
(MLE included only via its documented failure case), and save/load round-trips.

## Phase 3 — Prediction engine

A prefix trie for word completion, plus a unified `Predictor.predict()` serving
both modes, blending a personalization layer: a small per-user n-gram model
trained continuously on what the user types, interpolated with the base model
(`λ·P_user + (1−λ)·P_base`, λ growing as user data accumulates). Deliberately
skips `<UNK>`-ing the user's own vocabulary — capturing names and jargon the
base corpus would discard is the entire point.

**Deliverables:** latency benchmark (p95 target 20–50 ms), keystroke savings
ratio, hit@1 / hit@3 / hit@5 on held-out text.
→ `notebooks/03_prediction_engine.ipynb`

**Done when:** p95 latency is inside the debounce budget on the pruned model,
and the user profile measurably lifts hit@3 on text containing invented tokens.

## Phase 4 — HMM POS tagger, English

Implemented from scratch — transition matrix, emission matrix, initial
distribution, all Laplace-smoothed — with **log-space Viterbi** decoding.
`nltk.HiddenMarkovModelTrainer` used only as an optional correctness
cross-check, never in the shipped path. Unknown words fall back to suffix
heuristics (`-ing`/`-ed` → VERB, `-ly` → ADV, `-tion` → NOUN, capitalized →
proper noun).

**Deliverables:** accuracy on held-out set, confusion matrix, comparison
against a most-frequent-tag baseline and against NLTK's tagger, plus one
worked-by-hand Viterbi trellis for the report.
→ `notebooks/04_hmm_english.ipynb`

**Done when:** it beats the most-frequent-tag baseline on held-out data and
Viterbi output matches the hand-computed trellis exactly.

## Phase 5 — HMM tagger, regional language (Hindi)

The same `HMMTagger` class, **zero code changes** — only the corpus differs,
plus the Devanagari branch of the suffix heuristic. Devanagari normalized via
Unicode NFC (`indic-nlp-library` where available); danda (।/॥) treated as
sentence punctuation, not merged into the preceding token.

**Deliverables:** side-by-side accuracy table (English vs. Hindi), TTR and OOV
rate comparison, error analysis of the most-confused tag pairs.
→ `notebooks/05_hmm_regional.ipynb`

**Done when:** the identical class trains and decodes Hindi, and the accuracy
gap vs. English is explained by measured OOV rate rather than asserted.

## Phase 6 — POS-aware reranking

Wires Phases 4/5 into the Phase 3 predictor. Context is tagged by the HMM's own
`viterbi()`; each candidate gets a context-free "typical tag" guess; final score:

```
score = log P(word | context) + alpha * log P(tag(word) | tag_context)
```

**Deliverables:** hit@k with vs. without reranking — the core ablation —
reported honestly even when the effect is small or mixed at sample scale. A
near-zero delta on a small corpus is a legitimate finding, not a failed
experiment. → `notebooks/06_pos_reranking.ipynb`

**Done when:** `alpha = 0` reproduces Phase 3 rankings exactly, and the
ablation table is generated from held-out data.

## Phase 7 — Desktop app

Riskiest phase, done **last**, after the NLP core is provably working.

- **Tray icon** — `pystray` + Pillow: enable/disable, language toggle,
  settings, "retrain on my typing", quit.
- **Global keystroke capture** — `pynput.keyboard.Listener`, rolling context
  buffer reset on Enter / click / focus change, debounced ~50 ms.
- **Caret positioning** — the hard part. `GetGUIThreadInfo` via `ctypes` works
  for most Win32 apps; Electron apps and browsers often won't cooperate.
  Fallback: anchor near the mouse cursor.
- **Overlay window** — frameless, always-on-top, `WS_EX_NOACTIVATE` so it never
  steals focus (the bug that makes these apps unusable).
- **Injection** — `pynput` types the remaining characters on accept;
  clipboard+paste is faster for long completions but must save and restore the
  user's existing clipboard.

**Privacy — hard requirements, not options:** everything stays on the machine,
nothing is transmitted anywhere, the context buffer holds only the current
sentence and is never written to disk, and a password-field heuristic disables
capture entirely. Note for the demo: Windows Defender may flag a global
keyboard hook as a keylogger — develop inside an exclusion folder and disclose
this in the write-up.

**Fallback if the global hook proves unworkable:** a built-in Tkinter/Qt editor
window with live suggestions. Less impressive as a product, demonstrates every
NLP component correctly, and costs no lab marks.

**Done when:** suggestions appear and accept correctly in Notepad and in one
browser, without focus theft.

## Phase 8 — Evaluation, packaging, report

Bundle with PyInstaller into a single `.exe` with the pruned model embedded.
Final metrics table:

| Metric                        | What it shows                               |
| ----------------------------- | ------------------------------------------- |
| Perplexity                    | Language model quality, per order/smoothing |
| Keystroke savings ratio       | The headline product number                 |
| hit@1 / @3 / @5               | Prediction accuracy                         |
| p50 / p95 latency             | Real-time usability                         |
| Tagger accuracy               | English vs. regional, per tag               |
| Model size, RAM, startup time | Practicality                                |

---

## Sequencing

`0 → 1 → 2 → 3` first, so there is a working predictor with real numbers early.
Then `4 → 5 → 6` for the POS-tagging coursework. Phase 7 last — it is the phase
most likely to blow its estimate, and if the global hook doesn't pan out, the
editor-window fallback still delivers every required NLP component.

## Status

| Phase | Title                    | Status      |
| ----- | ------------------------ | ----------- |
| 0     | Scaffolding              | **done**    |
| 1     | Preprocessing            | **done**    |
| 2     | N-gram model             | **done**    |
| 3     | Prediction engine        | not started |
| 4     | HMM tagger, English      | not started |
| 5     | HMM tagger, Hindi        | not started |
| 6     | POS-aware reranking      | not started |
| 7     | Desktop app              | not started |
| 8     | Evaluation and packaging | not started |
