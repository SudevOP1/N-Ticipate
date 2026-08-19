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

Corpus: NLTK Brown, read via `.sents()`. Config as committed
(`min_token_freq: 2`, `max_vocab_size: 50000`, split 80/10/10, seed 42).

| Metric              | Raw corpus | After preprocessing (train) |
| ------------------- | ---------: | --------------------------: |
| Sentences           |     57,340 |                      45,610 |
| Tokens              |  1,161,192 |                     927,438 |
| Types               |     56,057 |                      24,634 |
| Type-token ratio    |     0.0483 |                      0.0266 |
| Hapax types         |     25,559 |                           0 |
| Mean sentence len   |      20.25 |                       20.33 |

- Splits: 45,610 train / 5,701 dev / 5,702 test.
- OOV rate on dev: **3.68%**.
- Truecase map: 49,221 entries.
- Vocabulary coverage: top 100 types = 55.3% of tokens, top 1,000 = 74.9%,
  top 5,000 = 89.9%, top 20,000 = 99.0%.
- Pipeline runtime: 4.8 s (Brown, end to end).
- Processed corpus: `data/processed/brown.json`, 11.8 MB.

Headline finding for the report: 5,000 types cover ~90% of running text while
25,559 types occur exactly once. That asymmetry is the direct justification for
the `<UNK>` cutoff here and for the n-gram pruning in Phase 2.

Hapax falls to zero after preprocessing by construction (every once-seen token
is now `<UNK>`); types halve while tokens barely move, which is the argument
for the `<UNK>` class in one line.

### Two bugs worth writing up

1. **Devanagari tokenisation.** A vowel sign (matra) such as U+093E is Unicode
   category `Mc` — a mark, not a letter — so both `\w` and `[^\W\d_]` reject
   it and `भारत` tokenised as `['भ', 'ा', 'रत']`. The word pattern now matches
   letters *and* combining marks. Caught by a Phase 1 test, four phases before
   Hindi was due.
2. **Brown's raw form is POS-tagged** (`The/at Fulton/np-tl`). Loading via
   `.raw()` would have trained the language model on tag-suffixed
   pseudo-words; the loader uses `.sents()` and a test asserts no token
   contains `/`.

Also: a membership set rebuilt inside a comprehension condition made
`build_vocab` O(types x vocab) — 270 s on Brown, 4.8 s after hoisting it.

Tests: 157 passing (`tests/test_phase0_setup.py`, `tests/test_preprocess.py`).
Deliverable notebook: `notebooks/01_preprocessing.ipynb` (executed, with Zipf
and coverage plots).

## Phase 2 — N-gram model

Trained on the Phase 1 Brown split (45,610 train sentences, vocab 24,634).
Perplexity measured on the dev split (5,701 sentences).

### Perplexity: order x smoothing

|   n | MLE      | Laplace | Stupid backoff\* | Kneser-Ney |
| --: | -------- | ------: | ---------------: | ---------: |
|   1 | 700.4    |   703.5 |            700.4 |      700.4 |
|   2 | infinite | 1,701.7 |            222.2 |      238.4 |
|   3 | infinite | 9,704.3 |            291.3 |  **221.1** |

\* Stupid backoff is not a probability distribution — measured mass over the
vocabulary after "the" is **1.306**, not 1.0. Its scores rank correctly, which
is all the app needs, but the number in that column is not a perplexity in the
same sense as the others. The model exposes this as `is_normalized` rather than
printing the figure silently.

N-gram counts: 24,636 unigrams, 360,323 bigrams, 1,050,690 trigrams.

Three findings worth the write-up:

1. **MLE is infinite for n > 1**, finite at n = 1. The unigram case is finite
   only because Phase 1 closed the vocabulary — every dev token is known or
   already `<UNK>`. Add context and unseen n-grams appear immediately.
2. **Laplace gets worse as order rises** (704 → 1,702 → 9,704). Add-1 hands a
   pseudo-count to each of ~24,600 vocabulary items for *every* context, and
   trigram contexts are mostly seen once or twice, so smoothing mass swamps
   the evidence. The textbook method is the worst performer at the order we
   want to ship.
3. **Only Kneser-Ney improves at n = 3** (238 → 221).

### Continuation counts (why KN works)

| word      | frequency | distinct predecessors |
| --------- | --------: | --------------------: |
| angeles   |        36 |                     1 |
| francisco |        30 |                     3 |
| york      |       252 |                     6 |
| time      |     1,284 |                   206 |
| said      |     1,567 |                   397 |

### Pruning: size vs. quality (trigram, dev perplexity)

| min_count | max_cont | n-grams   | size MB | Stupid backoff | Kneser-Ney |
| --------: | -------: | --------: | ------: | -------------: | ---------: |
|         1 |        0 | 1,050,690 |    14.7 |          291.3 |  **221.1** |
|         2 |       50 |   151,014 |     2.3 |      **502.7** |      680.1 |
|         2 |       20 |   138,953 |     2.2 |      **570.5** |      918.2 |
|         3 |       20 |    79,295 |     1.2 |      **664.7** |    1,281.9 |
|         5 |       10 |    47,205 |     0.8 |      **885.5** |    2,336.2 |

**The two rankings disagree, and that decides what ships.** Unpruned, KN wins
(221 vs 291). Pruned to a shippable size, KN loses and degrades roughly three
times faster. The cause is structural: KN's estimates are built on continuation
counts, and pruning deletes exactly the rare single-occurrence n-grams those
counts come from. Stupid backoff only ever needed raw counts plus a fallback
path, so it degrades gracefully.

Shipped model: trigram, stupid backoff, `min_count=2`, `max_continuations=50`.
14.7 MB → 2.26 MB (6.5x), 1,050,690 → 151,014 n-grams.
`data/models/ngram_trigram_pruned.pkl`.

### Latency

`candidates()` on the pruned model: **p50 0.20 ms, p95 0.36 ms** — two orders of
magnitude inside the 50 ms budget. Achieved by caching context totals; the
naive `sum(counter.values())` per query is O(continuations) and a context like
`("the",)` has thousands.

### Sample generations (stupid backoff, truecased)

- n=1: `Spirit lead were at ? in authority by then ( Amen still and board there`
- n=2: `however , you are scientific laboratories , and the skies , brass bar on those who takes`
- n=3: `Things like the epithets must reasonably be ascribed to the Gothic or <UNK> religion are concerned`

The jump from order 1 to 3 is the clearest qualitative evidence in the project
that the model learned structure: bag of words → local grammar → near-parsing
clauses.

### Bug found

`candidates()` ranked `<UNK>` **first** after `in the`. It is a frequent class
even though each member is rare, so suggesting it is nonsense — the app would
literally offer the user "<UNK>". Now excluded alongside the boundary markers.
After the fix: `in the` → world / first / United / same / past;
`i would like to` → think / be / know / do.

Tests: 226 passing, 1 skipped (MLE has no mass on an unseen context, so its
normalisation check is not applicable).
Deliverable notebook: `notebooks/02_ngram_models.ipynb` (executed, with the
perplexity-by-order and size-vs-quality plots).

## Phase 3 — Prediction engine

Built on the Phase 2 pruned trigram model (stupid backoff, 151,014 n-grams).
Evaluated on the Brown dev split.

Ranking is shared by both modes so there is one policy, not two that drift:

    score(w) = lambda * P_user(w | h) + (1 - lambda) * P_base(w | h)

Caveat stated plainly in the write-up: the base model uses stupid backoff,
whose scores are unnormalised, so this mixes two unnormalised scores. It is a
ranking heuristic, not a probabilistic mixture — fine for ranking, but not
something to write as `P(w|h)` without the qualification.

### Accuracy

| Evidence available          | hit@1 | hit@3 | hit@5 |
| --------------------------- | ----: | ----: | ----: |
| Next-word (0 chars typed)   | 16.7% | 29.0% | 34.8% |
| Completion, 1 char typed    | 34.6% | 48.2% | 53.0% |
| Completion, 2 chars typed   | 40.2% | 53.6% | 58.8% |
| Completion, 3 chars typed   | 46.1% | 65.5% | 74.3% |

10,105 next-word positions; 3,024–5,210 completion positions per prefix length.
`<UNK>` targets excluded — the predictor is designed never to suggest `<UNK>`,
so scoring those positions would report misses for reasons unrelated to model
quality.

### Keystroke savings — the headline product number

| Metric                    | Value  |
| ------------------------- | -----: |
| Words simulated           |  6,022 |
| Suggestion accepted       |  82.0% |
| Keystrokes typed          | 18,502 |
| Keystrokes without app    | 31,050 |
| **Savings**               | **40.4%** |

Simulation: typing a word costs one keystroke per character plus a space;
accepting costs characters-typed-so-far plus one accept key. A suggestion is
only taken when it actually saves a keystroke — accepting a 3-letter word after
typing 2 letters saves nothing and no real user would do it.

### Latency

| Mode       | mean | p50  | p95  | p99  |
| ---------- | ---: | ---: | ---: | ---: |
| Next-word  | 0.66 | 0.65 | 0.72 | 1.22 |
| Completion | 0.47 | 0.43 | 1.08 | 1.68 |

(milliseconds; budget is p95 <= 50 ms, so ~70x inside it)

Trie: 24,634 words in 68,106 nodes. Character-based, so Phase 5's Devanagari
works with no changes (`भार` → `भारत`, `भारतीय` verified in tests).

### Personalisation ablation

User text using vocabulary Brown has never seen (project names, jargon),
492 tokens observed, evaluated on 3 held-out sentences of the same kind:

|              | hit@1 | hit@3 | hit@5 | savings |
| ------------ | ----: | ----: | ----: | ------: |
| Base only    |  6.7% |  6.7% |  6.7% |   14.9% |
| Personalised | 33.3% | 46.7% | 60.0% |   67.3% |

Qualitative:

| Context      | Base                          | Personalised                |
| ------------ | ----------------------------- | --------------------------- |
| `vit`        | vital, vitality, vitally      | **viterbi**, vital, vitality |
| `ntic`       | *(nothing)*                   | **nticipate**               |
| `nticipate ` | the, `,`, `.`                 | uses, suggests, the         |

`ntic` returns **nothing** from the base model — the word is not in Brown, so no
amount of good language modelling can produce it. Only the profile can.

The non-obvious part: lambda is only **0.04** at 492 tokens, yet rankings flip
completely. Not a contradiction — for these words `P_base` is effectively zero,
so a small weight on a non-zero user score dominates. The interpolation does not
need to be aggressive to capture new vocabulary, which is why the cap can stay
conservative (0.4) and protect ordinary English.

### Bug found

`NgramModel.candidates()` walked the **entire** unigram table at the final
backoff level — 24,636 words scored per call. Hidden in Phase 2 because with
`k=3` the quota (`k*5`) was met by the higher orders; it only appeared when
Phase 3 raised the candidate pool to 50, at which point evaluation went from
seconds to over ten minutes. The last backoff level now reads a precomputed
top-500 unigram list, and the quota check moved inside the word loop.

### Defect recorded, not silently patched

Next-word suggestions after `is building` are `.`, `,`, `of`. The model is not
wrong — punctuation genuinely follows most often — but as a *suggestion* `.` is
useless: the user types it faster than they can read it. Keeping punctuation in
the language model remains correct (Phase 1's argument stands). This is a
display policy, so it is an opt-in flag (`prediction.suggest_punctuation`),
default on, which Phase 7 turns off. All metrics above were measured with the
filter off so they describe the model, not a filter over it.
Filtered: `is building` → of, a, on, and.

Tests: 322 passing, 1 skipped
(`test_trie.py`, `test_userprofile.py`, `test_predictor.py`).
Deliverable notebook: `notebooks/03_prediction_engine.ipynb` (executed, with
the accuracy-vs-evidence and lambda-growth plots).

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
