# N-Ticipate — running notes

Every number that ends up in the final write-up gets logged here as it is
measured, with the config that produced it. Phase 8 assembles the report from
this file rather than re-running everything at the deadline.

## Phase 0 — Scaffolding

- Python 3.12, Windows 11.
- `config.yaml` holds every tunable for all 9 phases; nothing is hardcoded.
- `tests/test_phase0_setup.py` asserts that every config key a later phase
  reads actually exists, so a typo surfaces now instead of halfway through a
  training run.

Environment status: run `python setup_env.py --check`.

## Phase 1 — Preprocessing

_(token counts before/after, Zipf plot, TTR, vocab coverage)_

## Phase 2 — N-gram model

_(perplexity table across order x smoothing, model sizes, pruning trade-off)_

## Phase 3 — Prediction engine

_(hit@1/3/5, keystroke savings, p50/p95 latency)_

## Phase 4 — HMM tagger, English

_(accuracy vs. most-frequent-tag baseline, confusion matrix, hand trellis)_

## Phase 5 — HMM tagger, Hindi

_(accuracy side-by-side with English, OOV rate, confused tag pairs)_

## Phase 6 — POS-aware reranking

_(the ablation: hit@k with vs. without reranking)_

## Phase 7 — Desktop app

_(apps tested, caret-positioning successes and failures)_

## Phase 8 — Packaging

_(exe size, RAM, startup time)_
