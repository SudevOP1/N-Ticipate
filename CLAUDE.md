# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

N-Ticipate is a from-scratch NLP autocomplete engine (n-gram LM + HMM POS tagger) that will ship as a Windows tray app suggesting text in any application. It is a college NLP lab project built in numbered phases. `PLAN.md` is the spec — read the relevant phase section before implementing it. `report/notes.md` is the running log of measured numbers for the write-up; add to it when a phase produces new metrics.

## Commands

The venv is at `.venv/` (Windows layout — `.venv/Scripts/python.exe`).

```bash
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
| `predictor.py` | 3, 6 | `Predictor.predict()` — the single entry point for both prediction modes |
| `hmm.py` | 4, 5 | `HMMTagger` — from-scratch transition/emission/initial matrices, log-space Viterbi |
| `app/` | 7 | Tray, hooks, overlay, injector — not yet written |

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

## Tests and notebooks

`tests/test_<module>.py`, one per source module, plus `test_phase0_setup.py` asserting the config contains every key later phases read. Notebooks under `notebooks/` are per-phase deliverables (tables and plots for the report) — the implementation always belongs in `nticipate/`, never in a notebook.

## Current state

Phases 0–5 are committed and the suite is green (448 passed, 1 skipped). Two things are stale: the status table at the bottom of `PLAN.md` still lists Phase 5 as not started, and `notebooks/05_hmm_regional.ipynb` plus the Phase 5 section of `report/notes.md` are unwritten. Phases 6–8 have not been started.
