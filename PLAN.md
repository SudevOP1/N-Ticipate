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
| Evaluation + report                  | 9       |

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
    __main__.py           Phase 7  entry point: tray | --editor | --check
    tray.py               Phase 7  tray icon + the object that wires it all up
    hooks.py              Phase 7  context buffer, key router, debouncer, hook
    overlay.py            Phase 7  the frameless, never-focusable popup
    injector.py           Phase 7  typing / pasting an accepted suggestion
    win32.py              Phase 7  ctypes: caret, window styles, clipboard
    editor.py             Phase 7  the fallback editor window
scripts/
  fetch_data.py           Phase 8  download + clean the corpora
  retrain.py              Phase 8  rebuild every shipped artefact
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

**Built.** Six modules plus an entry point; measured numbers in
`report/notes.md`. p95 prediction latency 5.4 ms against a 50 ms budget,
keystroke savings 27.4% end to end — both measured on the Brown-era model, so
Phase 8 re-measures them on the retrained one. Caret positioning confirmed working in
Notepad and confirmed unavailable in Chromium-class windows (mouse fallback).
The overlay's `WS_EX_NOACTIVATE` / `WS_EX_TOOLWINDOW` / `WS_EX_TOPMOST` styles
are asserted programmatically. The editor fallback was written anyway, because
it is the demo that cannot fail. Outstanding: the by-hand check in Notepad and
a browser with the hook actually running.

## Phase 8 — Corpus swap and retraining

Not in the original plan. Phases 0–6 were built on NLTK Brown (1961 edited
American prose, 1.16M tokens) with NLTK `treebank` / `indian` for the taggers,
and every metric came back green. Typing at the Phase 7 app showed what the
metrics could not: the suggestions were period-correct and unusable. `can you
pl` ranked `place, play, plan` and had no `please` at any rank; `Internet` was
not in the vocabulary at all.

The failure is one of evaluation, not of code. Perplexity and hit@k measured
on held-out text *from the same corpus* cannot see a domain mismatch, because
the held-out text shares it. That is the finding this phase contributes to the
report, and it is the argument for Phase 7 having been built at all: the app
is what exposed it.

This phase exists because Phase 7 shipped. It could not have been written
earlier — nothing before Phase 7 was capable of showing the problem.

**No module changes.** Only corpora, three config paths and the vocabulary /
pruning thresholds moved.

| Role            | Was                      | Now                                                    |
| --------------- | ------------------------ | ------------------------------------------------------ |
| Language model  | NLTK Brown               | `HuggingFaceFW/fineweb-edu` (`sample-10BT`, streamed) + `knkarthick/dialogsum` |
| Tagged English  | NLTK `treebank` (WSJ)    | UD English-EWT + UD English-GUM                        |
| Tagged Hindi    | NLTK `indian` (~10k tok) | UD Hindi-HDTB                                          |

fineweb-edu is modern web prose already quality-filtered and deduplicated, so
the cleaning left is line-level; dialogsum supplies the messenger register the
app actually runs in. DailyDialog would have been the better-known choice but
every Hub copy is script-based and `datasets` 5.x no longer executes loading
scripts — same reason the tagged corpora come from the UD GitHub repos as
`.conllu` rather than from the Hub's `universal_dependencies`.

Two scripts, both outside `nticipate/` because nothing the app imports at
runtime may depend on them: `scripts/fetch_data.py` downloads and cleans,
`scripts/retrain.py` re-runs Phases 1, 2, 4 and 5 and writes every artefact.

**Measured so far** (full tables in `report/notes.md`):

| Metric                     |  Brown |  Modern |
| -------------------------- | -----: | ------: |
| Train tokens               |   927k |   6.73M |
| OOV rate on dev            |  3.68% |   2.07% |
| Next-word hit@1 / @3       | 16.7% / 29.0% | 19.2% / 32.1% |
| Completion hit@1 (2 chars) |  40.2% |   46.0% |
| Keystroke savings          |  40.4% |   42.7% |
| p95 latency, completion    | 1.08 ms | 2.15 ms |
| Hindi tagger               | never built | 94.72% (baseline 92.26%) |

### Still to do

1. **Point the Phase 7 app at the new artefacts and confirm it uses them.**
   `config.yaml` already names them — `app.models.corpus` is
   `data/processed/modern.json`, `app.models.ngram` is the retrained
   `ngram_trigram_pruned.pkl`, and `app.models.tagger_hindi` now resolves to a
   file that exists for the first time. `--check` reports all four present,
   but the tray has not been run against them, so the end-to-end numbers in
   `report/notes.md` for Phase 7 (p95 5.4 ms, keystroke savings 27.4%) are
   still the Brown-era ones and `NticipateApp.warmup()` has never been timed
   on a model 5.5x larger.
2. **Re-derive every tunable that was swept on Brown.** `reranking.alpha =
   0.1` was chosen on 400 held-out *Brown* sentences, `hmm.smoothing_k = 0.01`
   on *treebank*, and the pruning pair (`min_count=2`,
   `max_continuations=50`) on Brown's size-vs-perplexity curve. All three are
   now unjustified numbers that happen to still be in the file.
   `preprocessing.min_token_freq = 3` and `max_vocab_size = 80000` were set by
   judgement during the swap and have never been swept at all.
3. **Re-measure `typical_tag_agreement()`.** The 93% figure that justifies the
   context-free candidate tag was measured on Brown + treebank. Phase 6's
   central shortcut rests on it.
4. **Re-run the Phase 2 and Phase 3 tables** — perplexity by order x
   smoothing, the pruning trade-off, the personalisation ablation — on the new
   corpus.
5. **Regenerate the notebooks.** 01–04 and 06 still print Brown-era tables;
   `05_hmm_regional.ipynb` was never written and now has a real Hindi model to
   write about.
6. **Fix the startup regression the swap introduced.** `data/processed/`
   `modern.json` is 77 MB and the app loads it for the truecase map alone —
   ~2.4 s of a ~3.1 s model load. A truecase-only artefact takes it back under
   a second.
7. **Delete `data/processed/brown.json`.** 29 MB, no longer read by anything.
8. **Exercise the Hindi path end to end.** `hmm_hindi.pkl` exists for the
   first time, so the tray's language toggle has never actually been run
   against a model rather than against its own refusal branch.

**Deliverables:** the before/after tables above, the re-swept parameter values
with the curves that justify them, and the regenerated notebooks.

**Done when:** the tray app runs on the retrained models with its latency and
keystroke-savings numbers re-measured, every number in `report/notes.md` was
produced by the current corpora, and no value in `config.yaml` is still
justified by a Brown-era sweep.

## Phase 9 — Evaluation, packaging, report

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
Then `4 → 5 → 6` for the POS-tagging coursework. Phase 7 next — it is the phase
most likely to blow its estimate, and if the global hook doesn't pan out, the
editor-window fallback still delivers every required NLP component.

Phase 8 was not in the original plan. It exists because Phase 7 shipped: only
a running app could show that the corpus, not the code, was the limit. Phase 9
depends on 8 closing, since the final metrics table cannot be assembled from a
mixture of Brown-era and current numbers.

## Status

| Phase | Title                       | Status          |
| ----- | --------------------------- | --------------- |
| 0     | Scaffolding                 | **done**        |
| 1     | Preprocessing               | **done**        |
| 2     | N-gram model                | **done**        |
| 3     | Prediction engine           | **done**        |
| 4     | HMM tagger, English         | **done**        |
| 5     | HMM tagger, Hindi           | **done**        |
| 6     | POS-aware reranking         | **done**        |
| 7     | Desktop app                 | **done**        |
| 8     | Corpus swap and retraining  | **in progress** |
| 9     | Evaluation and packaging    | not started     |

Phase 7's one open item is the by-hand check in Notepad and a browser with the
hook actually running. Phase 8's data swap, retraining and measurement are
done; the eight items under "Still to do" in that section are what is open —
the first of them is running Phase 7's app on the retrained models.
