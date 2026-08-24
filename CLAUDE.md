# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

N-Ticipate is a from-scratch NLP autocomplete engine (n-gram LM + HMM POS tagger) that will ship as a Windows tray app suggesting text in any application. It is a college NLP lab project built in numbered phases. `PLAN.md` is the spec — read the relevant phase section before implementing it. `report/notes.md` is the running log of measured numbers for the write-up; add to it when a phase produces new metrics.

## Commands

The venv is at `.venv/` (Windows layout — `.venv/Scripts/python.exe`).

```bash
.venv/Scripts/python.exe -m nticipate.app --check            # what the desktop layer can do here
.venv/Scripts/python.exe -m nticipate.app                    # tray icon + global hook
.venv/Scripts/python.exe -m nticipate.app --editor           # fallback editor window

.venv/Scripts/python.exe -m pytest -q                        # full suite (~25s)
.venv/Scripts/python.exe -m pytest tests/test_hmm.py -q      # one module
.venv/Scripts/python.exe -m pytest tests/test_hmm.py::test_name -q
.venv/Scripts/python.exe -m pytest -k viterbi -q             # by keyword

python setup_env.py            # create data dirs + download NLTK corpora
python setup_env.py --check    # report status only, download nothing
```

There is no pytest config file, no linter, and no build step. Run pytest and notebooks from the project root so `import nticipate` resolves.

## Architecture

One module per phase, all under `nticipate/`:

| Module | Phase | Role |
| --- | --- | --- |
| `config.py` | 0 | Cached YAML loader — `load_config()`, `get("dotted.key", default)`, `require()`, `resolve_path()`, `data_dir()` |
| `preprocess.py` | 1 | `clean_text → segment_sentences → tokenize → build_vocab/apply_unk → truecase → pad_sentence → train_dev_test_split`, bundled as `Corpus` |
| `ngram.py` | 2 | `NgramModel` — MLE / Laplace / stupid-backoff / Kneser-Ney, perplexity, pruning, generation |
| `trie.py`, `userprofile.py` | 3 | Prefix trie for completions; per-user n-gram model learned from typing |
| `predictor.py` | 3, 6 | `Predictor.predict()` — the single entry point for both prediction modes; attaching an `HMMTagger` turns on the Phase 6 POS term |
| `hmm.py` | 4, 5 | `HMMTagger` — from-scratch transition/emission/initial matrices, log-space Viterbi |
| `app/hooks.py` | 7 | `ContextBuffer`, `KeyRouter`, `Debouncer`, `CapturePolicy`, `KeystrokeHook` |
| `app/overlay.py` | 7 | `SuggestionOverlay` — frameless, never-focusable Tk popup |
| `app/injector.py` | 7 | `plan_injection()` + `Injector` — types or pastes an accepted word |
| `app/win32.py` | 7 | ctypes: caret rect, window styles, clipboard, password detection |
| `app/tray.py` | 7 | `NticipateApp` (the wiring) + `TrayIcon` (pystray) |
| `app/editor.py` | 7 | `EditorApp` — the Tk fallback window |

Data flow: raw text → `Corpus` (JSON in `data/processed/`) → `NgramModel` (pickle in `data/models/`) → `Trie` + `UserProfile` → `Predictor`. `HMMTagger` trains on a separate tagged corpus (`.conll` in `data/raw/`, or NLTK `treebank`/`indian`) and joins the pipeline in Phase 6, where it reranks the predictor's candidates.

Everything under `data/models/` and `data/processed/` is gitignored — those artifacts are rebuilt by `setup_env.py` and the notebooks, not committed.

### Config is the only place for tunables

No module hardcodes a magic number. Every parameter lives in `config.yaml`, sectioned by phase, and is read through `nticipate.config.get()`. Functions and constructors take explicit kwargs that default to `None` and fall back to the config value, so callers (tests, notebooks) can override without touching the file. When adding a parameter, add it to `config.yaml` with a comment explaining *why* that value, not just what it does.

### Persistence

`NgramModel` and `HMMTagger` use `to_dict()`/`from_dict()` + pickle; `Corpus` and `UserProfile` use JSON (`ensure_ascii=False` — the Hindi corpus depends on it). Any change to a model's internals must keep the dict round-trip working; tests assert it.

## Deliberate decisions that look like bugs

These are documented in module docstrings and measured in `report/notes.md`. Do not "fix" them without reading the reasoning first.

- **Stopwords and punctuation are kept.** Removing them is right for classification and wrong for language modelling — `of the` is exactly what a bigram model must predict.
- **Counting is lowercased; output is truecased.** `build_truecase_map` restores natural casing at suggestion time so counts aren't split across surface forms.
- **`hmm.smoothing_k = 0.01`, not 1.0.** Add-one puts more mass on the smoothing term than on the counts at this vocab size and drops the tagger below the most-frequent-tag baseline. Swept in notebook 04.
- **`NEG_INF = -1e30`, not `-math.inf`.** NumPy warns on `inf - inf` inside the Viterbi max; a very negative finite number ranks identically.
- **Linear-space Viterbi is implemented but unused.** It exists to demonstrate underflow (`Trellis.underflowed`) for the report.
- **NLTK's HMM appears only in `nltk_cross_check()`.** It is a correctness check, never on the shipped prediction path. Same for the NLTK tokenizer/splitter, which have regex fallbacks so Phase 1 works before corpora are downloaded.
- **Stupid backoff is unnormalised**, so the predictor's `λ·P_user + (1−λ)·P_base` blend mixes two unnormalised scores. It is a ranking heuristic, not a probability — don't write `P(w|h)` about it in the report without the caveat.
- **The user profile's vocabulary is never `<UNK>`-ed.** Capturing names and jargon the base corpus discards is the entire point of personalization.
- **Phase 5 adds no new class.** Hindi runs through the same `HMMTagger` with only a different corpus plus the Devanagari branch of the suffix heuristic; its tests live in `tests/test_hmm.py`.
- **Phase 6 adds no new class either.** The reranker is a term inside `Predictor.predict()`; its tests live in `tests/test_predictor.py`. With no tagger attached the predictor is exactly Phase 3, which is why `reranking.enabled: true` is safe as a default.
- **Reranking scores are logs; Phase 3 scores are probabilities.** `predict()` switches representation when the POS term is live. The ranking is unaffected at `alpha = 0` (the log is monotonic) and a test asserts it.
- **The candidate's tag is guessed context-free.** Tagging each candidate in context would cost one Viterbi decode per candidate per keystroke. The shortcut agrees with the in-context tag 93% of the time — measured, not assumed, by `typical_tag_agreement()`.
- **The Phase 7 caret query returning `None` is the normal case, not a failure.** `GetGUIThreadInfo` answers for classic Win32 controls and tells browsers and Electron nothing, so `win32.caret_rect()` returns `None` and the overlay anchors to the mouse. The opaque window classes are listed explicitly so the API's stale rectangle is never trusted.
- **Learning is off by default (`app.learning.enabled: false`).** It is the one feature that writes what the user typed to disk, and Phase 7's privacy rule says the context buffer never reaches disk. Both hold only if learning is opt-in from the tray menu — and even then what is saved is the profile's n-gram counts, not the keystroke stream.
- **`ContextBuffer` has no `save`/`to_dict` and its `__repr__` prints lengths.** Privacy is structural here rather than a setting that is checked; tests assert both.
- **`NticipateApp.warmup()` is not a micro-optimisation.** The first prediction is ~360 ms (NLTK tokenizer load + the model's lazy index) against ~1.4 ms steady state. Without it the app appears to miss its latency budget by 10× on the first keystroke.
- **`reranking.tag_context_size` is a tagging window, not a tag order.** The HMM is a tag bigram, so exactly one preceding tag ever conditions the term. Raising the window above 2 cannot change anything in the running app: `predict()` has already trimmed the context to the trigram model's two words.

## Tests and notebooks

`tests/test_<module>.py`, one per source module, plus `test_phase0_setup.py` asserting the config contains every key later phases read. Notebooks under `notebooks/` are per-phase deliverables (tables and plots for the report) — the implementation always belongs in `nticipate/`, never in a notebook.

## Current state

Phases 0–7 are done and the suite is green (594 passed, 1 skipped). Two things are stale: `notebooks/05_hmm_regional.ipynb` and the Phase 5 section of `report/notes.md` were never written (the Phase 5 code and tests are complete), and `data/models/hmm_hindi.pkl` was never built — so the tray's Hindi language toggle refuses and logs a warning. Phase 8 has not been started.

Phase 7's remaining manual step: run the tray app and confirm by hand that suggestions appear and accept in Notepad and in a browser without the caret moving. Everything else in that phase is measured in `report/notes.md`.
