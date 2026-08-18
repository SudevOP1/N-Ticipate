# N-Ticipate

An n-gram + HMM powered text auto-completion engine, packaged as a
system-tray desktop app for Windows. Built in phases for a college NLP
lab — see `notebooks/` for the write-up of each experiment.

## Setup (Phase 0)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
python setup_env.py     # downloads NLTK corpora, checks config.yaml, creates data folders
pytest tests/ -v         # should pass right now -- these are scaffolding smoke tests
```

## Project map

```
n-ticipate/
├── config.yaml              all tunable parameters for every phase
├── setup_env.py             one-time environment + corpus setup
├── data/
│   ├── raw/                 downloaded corpora
│   ├── processed/           tokenized/split data (gitignored, regenerate via notebooks)
│   └── models/              trained n-gram / HMM models (gitignored)
├── nticipate/
│   ├── config.py            shared config loader
│   ├── preprocess.py        Phase 1
│   ├── ngram.py             Phase 2
│   ├── trie.py               Phase 3
│   ├── hmm.py                 Phases 4 & 5
│   ├── predictor.py          Phases 3 & 6
│   ├── userprofile.py        Phase 3 (personalization)
│   └── app/                   Phase 7 (tray, hooks, overlay, injector)
├── notebooks/                one notebook per lab experiment
├── tests/                     pytest suite, one file per module
└── report/                    write-up notes and figures
```

## Status

- [x] Phase 0 — scaffolding
- [x] Phase 1 — preprocessing
- [x] Phase 2 — n-gram models
- [x] Phase 3 — prediction engine
- [x] Phase 4 — HMM tagger (English)
- [x] Phase 5 — HMM tagger (regional language)
- [x] Phase 6 — POS-aware reranking
- [ ] Phase 7 — desktop app
- [ ] Phase 8 — evaluation, packaging, report

## Design notes

Everything runs locally — no network calls, nothing leaves the machine.
See `report/notes.md` for the running log of decisions worth justifying
in the final report (e.g. why stopwords are kept, not stripped).
