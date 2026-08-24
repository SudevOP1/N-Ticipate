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

## Corpus swap — Brown → modern web + dialogue

Phases 0–7 were built on NLTK Brown (1961, edited American prose, 1.16M
tokens) and NLTK `treebank` / `indian` for the taggers. Every metric was
green and the app still felt wrong to use, which is the whole reason this
section exists: perplexity and hit@k were measuring how well the model
predicted *Brown*, and that was never the question. Typing at it turned up
suggestions no 2026 user would accept — `can you pl` ranked `place, play,
plan` and had no `please` at any rank, and `Internet` was not in the
vocabulary. The measured before/after is at the end of this section.

The NLP code is unchanged. Only the corpora, three config paths and the
pruning/vocab thresholds moved. `scripts/fetch_data.py` downloads and cleans;
`scripts/retrain.py` re-runs Phases 1, 2, 4 and 5 and writes the artefacts.

| Role                | Was                       | Now                                              |
| ------------------- | ------------------------- | ------------------------------------------------ |
| Language model      | NLTK Brown                | HuggingFaceFW/fineweb-edu (`sample-10BT`, streamed) + knkarthick/dialogsum |
| Tagged English      | NLTK `treebank` (WSJ)     | UD English-EWT + UD English-GUM (web, blog, review, email) |
| Tagged Hindi        | NLTK `indian` (~10k tok)  | UD Hindi-HDTB                                    |

Why those three:

- **fineweb-edu** is modern web prose that has already been quality-filtered
  and deduplicated, so the cleaning left to do is line-level, not
  document-level. Streamed rather than downloaded — one shard is 2.15 GB and
  the run needs ~6M tokens of it.
- **dialogsum** is messenger-style dialogue, plain CSV, no loading script. It
  supplies the register the app actually runs in and that fineweb has almost
  none of: `let me know`, `see you tomorrow`, `can you please`. DailyDialog
  would have been the better-known choice, but every copy of it on the Hub is
  script-based and `datasets` 5.x no longer executes loading scripts.
- **Universal Dependencies** for the taggers, taken from the UD GitHub repos
  as `.conllu` rather than the Hub, because the Hub's `universal_dependencies`
  is also script-based. EWT/GUM is the domain the app runs in; HDTB is 30x the
  Hindi data NLTK `indian` had, which is what finally made
  `data/models/hmm_hindi.pkl` worth building.

### Phase 1 — preprocessing, before and after

| Metric              | Brown (train) | Modern (train) |
| ------------------- | ------------: | -------------: |
| Sentences           |        45,610 |        400,836 |
| Tokens              |       927,438 |      6,727,692 |
| Types               |        24,634 |         51,028 |
| Type-token ratio    |        0.0266 |         0.0076 |
| OOV rate on dev     |         3.68% |      **2.07%** |
| Processed corpus    |       11.8 MB |        77.1 MB |
| Pipeline runtime    |         4.8 s |         56.6 s |

`min_token_freq` went 2 → 3 and `max_vocab_size` 50,000 → 80,000 with the
swap. On a web corpus the hapax tail is mostly typos and scrape residue, and
every one of those was a word the trie could offer; the extra vocabulary
headroom is for real words the larger corpus actually supports.

The OOV rate falling from 3.68% to 2.07% on 7x the data is the number that
predicts the qualitative change: a third fewer of the words a user types are
ones the model has never seen.

### Phase 2 — n-gram model

| Metric                      |   Brown |   Modern |
| --------------------------- | ------: | -------: |
| n-grams before pruning      | 1,050,690 | 5,278,151 |
| n-grams after pruning       | 151,014 |  862,252 |
| Shipped model size          | 2.26 MB |  12.5 MB |
| Dev perplexity (pruned)     |   502.7 |    320.2 |

The perplexities are not directly comparable — different test sets — but they
are both "trigram, stupid backoff, pruned, measured on held-out text from the
same corpus", so the pair still says the modern model is not paying for its
larger vocabulary with a worse fit.

### Phase 3 — prediction accuracy, before and after

Held-out test sentences, 600-sentence sample, personalisation off, reranking
on at α = 0.1.

| Evidence available        | hit@1 (Brown → modern) | hit@3 | hit@5 |
| ------------------------- | ---------------------: | ----: | ----: |
| Next-word (0 chars)       |    16.7% → **19.2%**   | 29.0% → **32.1%** | 34.8% → **38.1%** |
| Completion, 1 char        |    34.6% → **37.2%**   | 48.2% → **51.0%** | 53.0% → **55.3%** |
| Completion, 2 chars       |    40.2% → **46.0%**   | 53.6% → **57.9%** | 58.8% → **63.4%** |
| Completion, 3 chars       |    46.1% → **51.8%**   | 65.5% → **70.0%** | 74.3% → **77.3%** |

Keystroke savings: **40.4% → 42.7%** (4,748 words simulated, acceptance rate
80.5%).

Latency, still far inside the 50 ms budget but no longer free — the model is
5.5x larger:

| Mode       | p50 (Brown → modern) | p95 |
| ---------- | -------------------: | --: |
| Next-word  |    0.65 → 0.70 ms    | 0.72 → 0.86 ms |
| Completion |    0.43 → 0.73 ms    | 1.08 → 2.15 ms |

### What the swap actually changed, side by side

Both rows below are the same predictor with the same UD-trained tagger
attached (α = 0.1, personalisation off); only the n-gram model and its corpus
differ. The Brown model was rebuilt for this table rather than quoted from
memory.

| Context + prefix  | Brown model                          | Modern model                            |
| ----------------- | ------------------------------------ | --------------------------------------- |
| `let me` + `k`    | know, knew, kind, known, keep        | know, known, keep, kind, King            |
| `see you` + `tom` | tomorrow, Tom, Tommy, tomb, Tom's    | tomorrow, Tom, tomatoes, tomato, tomb    |
| `can you` + `pl`  | place, play, plan, placed, planning  | **please**, place, play, plants, plant   |
| `how are` + ``    | you, not, the, in, a                 | you, your, things, they, not             |
| `thank` + `y`     | you, your, years, year, yet          | you, your, years, year, yes              |
| `talk to` + `y`   | you, years, your, year, yet          | you, your, years, year, yes              |
| `on the` + ``     | other, basis, ground, floor, part    | other, **Internet**, ground, basis, surface |

Worth being precise about this, because the honest version is less dramatic
than the complaint that prompted the swap. Brown is not incoherent — it gets
`let me know` and `see you tomorrow` right, and on those two rows the
difference is nil. What it does not have is:

1. **Conversational fixed phrases.** `can you please` does not occur in 1961
   news and fiction, so `please` is not a candidate at any rank. This is the
   clearest single case and it is the shape of the failure the app was showing.
2. **Modern vocabulary.** `on the Internet` is the second-ranked continuation
   in the new model and is not in Brown's vocabulary at all.
3. **Useful mass in the tail.** Brown fills ranks 2–5 of `how are ___` with
   `not, the, in, a` — function words that would never be worth accepting.
   The modern model spends the same slots on `your, things, they`.

The lesson for the write-up is about evaluation, not about Brown: hit@k and
perplexity measured on held-out text *from the same corpus* cannot see a
domain mismatch, because the held-out text shares it. Only running the app
against real typing exposed it, which is an argument for Phase 7 existing at
all.

### Cost paid

Startup grew: `Corpus.load()` on a 77.1 MB JSON is ~2.4 s of the app's ~3.1 s
model load. The app reads that file for the truecase map alone.

**Fixed** (truecase-only artefact). `Corpus.save_truecase()` writes the map on
its own to `data/models/truecase.json`; `load_predictor()` prefers it and falls
back to the corpus only when it is missing.

| Source of the truecase map | Size | Parse | Full `load_predictor()` |
| --- | --- | --- | --- |
| `data/processed/modern.json` | 77.1 MB | 9.0 s cold | `MemoryError` |
| `data/models/truecase.json` | 3.9 MB | 0.08 s | 1.69 s |

The `MemoryError` is not a footnote: with the 12.5 MB n-gram pickle and both
taggers already resident, decoding the 77 MB file exhausted the heap and the
app would not start at all. The two maps compare equal (153,955 entries), so
the suggestion surface forms are unchanged.

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

Corpus: NLTK Penn Treebank sample, universal (12-tag) tagset. 3,914 sentences,
100,676 tokens. Split 80/20, seed 42 → 3,131 train / 783 test (19,655 test
tokens). Model: bigram HMM, Laplace-smoothed initial / transition / emission
matrices, log-space Viterbi, all from scratch.

The universal tagset rather than the full 45-tag Penn set is deliberate: Phase
6 only needs noun-ish vs. verb-ish to rerank, and 45 tags would spread the same
training data over nearly four times the transition parameters.

### Headline accuracy

| Tagger                    |    all |   seen | unseen |
| ------------------------- | -----: | -----: | -----: |
| **HMM (Viterbi)**         | **95.39%** | 96.98% | 74.77% |
| Most-frequent-tag baseline | 92.74% | 96.21% | 47.83% |

OOV rate on the test set: **7.16%**. Improvement over baseline: +2.65 points,
which is a 36.5% cut in error rate. Both taggers use the same definition of
"seen" (exact form or lowercased form in training) so the unseen columns
compare like for like.

Nearly all of the gain is on unseen words — the baseline is very hard to beat
on words it has memorised, which is the honest reading of that table.

### The smoothing k finding — worth a paragraph in the write-up

Add-one is the textbook default and is measurably the wrong value here. With
~11k vocabulary types the `k·(V+1)` term dominates the emission denominator,
flattens every row, and the from-scratch HMM ends up *below* the baseline it is
supposed to beat.

|      k | accuracy |  seen | unseen |
| -----: | -------: | ----: | -----: |
|    1.0 |   91.43% | 92.73% | 74.48% |
|    0.1 |   95.18% | 96.76% | 74.70% |
| **0.01** | **95.39%** | 96.98% | 74.77% |
|  0.001 |   95.41% | 97.00% | 74.77% |
|  1e-05 |   95.41% | 97.00% | 74.77% |

Shipped at `k = 0.01` (`hmm.smoothing_k`), on the plateau. The result looks
exactly like a Viterbi bug and is not one — worth saying, because that is the
first place anyone would look.

### Unseen words: prior × strategy ablation

Two orthogonal knobs. The **prior** is what an unseen word looks like before
its spelling is considered (`hapax` estimates `P(tag | unseen)` from words seen
exactly once, then divides out `P(tag)` by Bayes; `laplace` uses the smoothing
floor). The **strategy** is what the spelling adds (`suffix` morphology,
`most_frequent_tag`, or `uniform` = nothing).

| prior   | strategy          | accuracy | unseen |
| ------- | ----------------- | -------: | -----: |
| hapax   | **suffix**        | **95.39%** | **74.77%** |
| hapax   | uniform           |   94.52% | 62.69% |
| hapax   | most_frequent_tag |   93.96% | 55.15% |
| laplace | suffix            |   94.98% | 69.23% |
| laplace | uniform           |   92.95% | 40.51% |
| laplace | most_frequent_tag |   94.05% | 56.08% |

Reading: the hapax prior alone takes unseen-word accuracy 40.5% → 62.7%; the
suffix rules alone take it 40.5% → 69.2%; together 74.8%. The prior is the
cheaper of the two (a count, no linguistic knowledge) and transfers to Phase 5
unchanged, while the suffix rules are the part that needs a Devanagari branch.

`seen` accuracy is identical (96.95–97.00%) across all six rows, as it must be
— both knobs only touch out-of-vocabulary words. A column that moved there
would mean a knob was leaking into the rest of the model.

### Log space is not optional

Linear-space Viterbi is implemented alongside the log-space one purely to
measure this. On Treebank text the best-path probability drops below float64's
smallest normal and hits exactly `0.0` at **125 tokens**. Past that point every
path scores zero, `argmax` returns index 0, and the tagger emits one tag for
the whole sentence — a confidently wrong answer, not a degraded one. A
paragraph pasted into the Phase 7 app is well past that limit.

### Correctness

Viterbi is checked against **exhaustive enumeration** of every tag sequence on
short inputs (27 sequences for a 3-token, 3-tag case), in both the notebook and
`tests/test_hmm.py`. That is the real check.

NLTK's `HiddenMarkovModelTrainer`, trained on the identical split and evaluated
on 200 OOV-free test sentences (3,728 tokens): **90.08%**, against **96.94%**
for this implementation on the same sentences. Report this carefully — NLTK's
`train_supervised` defaults to a near-unsmoothed emission estimator, so the gap
says something about its defaults and nothing about whether this decoder is
correct. Restricting to OOV-free sentences at least isolates the decoder from
the unknown-word policy.

### Per-tag accuracy and errors

| Tag  | Accuracy |     n |
| ---- | -------: | ----: |
| ADJ  |   0.846  | 1,256 |
| ADV  |   0.901  |   648 |
| VERB |   0.931  | 2,676 |
| PRT  |   0.942  |   639 |
| NUM  |   0.947  |   647 |
| X    |   0.948  | 1,263 |
| DET  |   0.964  | 1,699 |
| NOUN |   0.965  | 5,665 |
| ADP  |   0.967  | 1,933 |
| CONJ |   0.998  |   434 |
| PRON |   0.998  |   535 |
| .    |   1.000  | 2,260 |

Most common confusions: VERB→NOUN (133), ADJ→NOUN (98), NOUN→ADJ (97),
NOUN→VERB (88), DET→ADP (43).

The errors sit exactly where English is genuinely ambiguous. ADJ/NOUN is
noun-modifying-noun ("the **oil** price") against true adjectives; NOUN/VERB is
the `-ing`/`-ed` forms and the large class of words that are both ("the
**runs**" vs. "he **runs**"). Closed-class tags are essentially solved, being
short lists of unambiguous frequent words.

Visible in the demo output: *The quick brown fox **jumps** over…* tags `jumps`
as NOUN. It is a known word, so no heuristic is at fault — noun-noun sequences
are common enough in newspaper text that the transitions prefer one there.

Implication for Phase 6, stated before the ablation is run: the reranker will
be most confident exactly where tags are unambiguous, and that is where it adds
least. The cases where it could help most are the ones where it is least
reliable. A small or mixed effect in Phase 6 would be consistent with this, not
a surprise.

### Cost

| Measure           | Value |
| ----------------- | ----: |
| Training          | 0.06 s over 3,131 sentences |
| Decoding          | 0.007 ms/token, 0.17 ms/sentence |
| Model on disk     | 1.17 MB (12 tags × 10,983 words) |

Phase 6 tags only the two or three words in front of the caret, so the
per-sentence figure is a generous upper bound; the reranker will barely touch
the ~50 ms keystroke budget.

Artefacts: `data/models/hmm_english.pkl`, and
`data/raw/sample_tagged_english.conll` (200 sentences, committed) so a fresh
clone with no NLTK downloads still runs.

Tests: 412 passing, 1 skipped (`tests/test_hmm.py` adds 89).
Deliverable notebook: `notebooks/04_hmm_english.ipynb` (executed — transition
heatmap, hand trellis, underflow curve, k sweep, ablation, confusion matrix).

## Phase 5 — HMM tagger, Hindi

Corpus: UD Hindi-HDTB (train + dev), converted to two-column `word<TAB>UPOS`
by `scripts/fetch_data.py`. English is UD English-EWT (train + dev) plus UD
English-GUM, same conversion. Both replaced the NLTK corpora — `indian` is
~10k Hindi tokens, which was too small to say anything, and `treebank` is WSJ
newswire, the wrong domain for an app that runs inside a browser.

`HMMTagger` is unchanged between the two rows below. Only the corpus differs,
plus the Devanagari branch of the suffix heuristic.

| Metric                     |    English |     Hindi |
| -------------------------- | ---------: | --------: |
| Sentences                  |     25,856 |    14,965 |
| Tokens                     |    429,831 |   316,274 |
| Types                      |     31,333 |    17,993 |
| Type-token ratio           |     0.0729 |    0.0569 |
| Tags in the tagset         |         17 |        16 |
| Train / test sentences     | 20,684 / 5,172 | 11,972 / 2,993 |
| **Accuracy**               | **93.48%** | **94.72%** |
| — on known words           |     94.32% |    95.50% |
| — on unseen words          |     70.66% |    67.86% |
| OOV rate (held-out)        |      3.54% |     2.81% |
| Most-frequent-tag baseline |     88.76% |    92.26% |
| Model size                 |     4.1 MB |    2.4 MB |
| Fit time                   |      2.3 s |     1.6 s |

Both beat their baseline, which is Phase 5's acceptance criterion.

### Hindi scores *higher* than English, and the OOV rate says why

This is the opposite of the result the plan expected, and the explanation is
the same one either way: the language with more of its held-out text unseen is
the one that scores lower. Here that is English — 3.54% OOV against Hindi's
2.81%, and TTR 0.073 against 0.057. HDTB is newswire with a narrow vocabulary;
EWT/GUM is email, blogs, reviews and forum posts, where the vocabulary is
wider and the spelling less consistent. English also has a harder tagset
distinction to make (see the confusion pairs below).

On the unseen words themselves the ordering flips back — English 70.66%
against Hindi 67.86% — which is what the suffix heuristic predicts: English
inflection is suffixal and shallow (`-ing`, `-ed`, `-ly`, `-tion`), Hindi
inflection is richer and more of it is carried by postpositions written as
separate tokens.

`tests/test_hmm.py::test_the_accuracy_gap_between_languages_tracks_the_oov_rate`
pins the *relationship* rather than the winner, so swapping either corpus
cannot quietly invalidate this paragraph.

### Most-confused tag pairs (gold → predicted, held-out)

| English            | count | Hindi              | count |
| ------------------ | ----: | ------------------ | ----: |
| PROPN → NOUN       |   483 | NOUN → PROPN       |   647 |
| VERB → AUX         |   365 | PROPN → NOUN       |   485 |
| VERB → NOUN        |   362 | VERB → AUX         |   289 |
| NOUN → PROPN       |   307 | ADP → ADV          |   228 |
| SCONJ → ADP        |   284 | NOUN → ADJ         |   194 |
| NOUN → VERB        |   274 | PROPN → ADJ        |   149 |

The NOUN/PROPN confusion dominates both languages and for different reasons.
English has a capitalisation cue that the tagger can use and that sentence
starts destroy; Devanagari has no case at all, so the only evidence for PROPN
is the transition matrix and whatever the word itself was seen as. That is
why the pair runs both directions in Hindi and is the single largest error
class there.

VERB → AUX is shared, and is a genuine tagset difficulty rather than a model
failure: UD splits the auxiliary off the main verb, so `है`/`है` and `have`
are the same surface word in both readings and only the context separates
them — exactly the case a tag bigram is weakest at.

## Phase 6 — POS-aware reranking

Wires the Phase 4 tagger into the Phase 3 predictor. No new class: the reranker
is a term inside `Predictor.predict()`, live only when a fitted `HMMTagger` is
attached.

```
score(w) = log[ λ·P_user(w | h) + (1−λ)·P_base(w | h) ]  +  α·log P(tag(w) | tag(h₋₁))
```

Setup for every number below: pruned trigram model (stupid backoff) from Phase
2, `hmm_english.pkl` from Phase 4, first 400 held-out Brown sentences — 7,888
next-word positions and 5,380 completion positions (2 characters typed).
Personalisation off, so the only thing moving between arms is the POS term.

Three pieces, each with a different cost:

| Piece | Source | Per keystroke |
| --- | --- | --- |
| context tag `tag(h₋₁)` | Viterbi over the last `tag_context_size` tokens | one 2-column trellis |
| candidate tag `tag(w)` | context-free `argmax_t P(t \| w)` | one emission lookup per candidate |
| the prior | the HMM's own transition matrix | one array index per candidate |

The candidate is tagged **without** context deliberately: tagging it properly
would mean one Viterbi decode per candidate, i.e. ~50 decodes per keystroke.

### The ablation — the core deliverable (α = 0.1)

| Mode | positions | | hit@1 | hit@3 | hit@5 |
| --- | ---: | --- | ---: | ---: | ---: |
| next word | 7,888 | rerank off | 16.13% | 29.06% | 35.17% |
| | | **rerank on** | **17.20%** | 28.98% | 35.07% |
| | | delta | **+1.08** | −0.08 | −0.10 |
| completion | 5,380 | rerank off | 41.69% | 55.61% | 62.03% |
| | | **rerank on** | **41.86%** | **55.87%** | **62.21%** |
| | | delta | +0.17 | +0.26 | +0.19 |

Keystroke savings, the headline product number: **40.45% → 40.57%**, i.e.
unchanged (6,383 → 6,380 words accepted from a suggestion out of 7,888).

The honest summary: **the POS term buys about a point of next-word hit@1 and a
quarter of a point at most anywhere else.** That is a real result and it is the
one the plan predicted might happen; it is not a failed experiment.

Why the shape makes sense. In next-word mode the model ranks the continuations
of a two-word context, many of them backed by thin trigram counts — the one
place a grammatical prior has room to break a tie. In completion mode two typed
characters have already narrowed the field far harder than "it should be a
noun" can. And next-word hit@3/@5 slipping a tenth of a point while hit@1 gains
a full one is the trade being made visible: reranking pulls grammatical words
up, and a few correct answers get pushed down with them.

### Choosing α

| α | hit@1 | hit@3 | hit@5 |
| ---: | ---: | ---: | ---: |
| 0.0 | 16.13% | 29.06% | 35.17% |
| 0.05 | 16.23% | 28.92% | 35.09% |
| **0.1** | **17.20%** | 28.98% | 35.07% |
| 0.2 | 17.18% | 29.01% | 34.91% |
| 0.3 | 17.09% | 29.20% | 34.44% |
| 0.5 | 16.71% | 28.68% | 33.91% |
| 1.0 | 15.57% | 27.09% | 32.24% |
| 2.0 | 14.36% | 24.99% | 29.27% |

hit@1 has a plateau over 0.1–0.3; hit@5 decays monotonically from the first
non-zero weight. `config.yaml` ships **α = 0.1**, the corner of the plateau: it
takes the whole hit@1 gain while leaving hit@3/@5 within a tenth of a point of
Phase 3. (0.3 was the initial guess in the config; the sweep is why it is not
the shipped value.)

α = 2 is worth keeping in the write-up as the failure mode — the tag term
overrules the language model and accuracy falls below the Phase 3 baseline at
every k. A reranker with too much authority stops being a reranker.

`α = 0` reproduces the Phase 3 ranking *exactly* (the log is monotonic), which
is asserted in the notebook and in `tests/test_predictor.py`. That is what
makes the two arms of the table above comparable.

### Bug found — the buffer has not ended

The obvious way to get the context tag is `tagger.viterbi(window)[-1]`. It is
wrong. `viterbi()` decodes a finished *sentence*, so it adds the transition
into the end-of-sentence state — and a buffer the user is still typing has not
ended.

That term is not small. The end state is reached almost only from punctuation:
on this tagger `log P(end | .) = −1.11` against `log P(end | VERB) = −13.90`,
a gap of nearly 13 nats against a trellis column that only favoured `VERB` over
`.` by about 9. So `i would` came back tagged `PRON .` — the model preferring
to believe `would` was a full stop over believing the sentence continued — and
every suggestion after a verb was scored against the transitions *out of
punctuation*.

The fix is to take the arg-max of the last trellis column and never apply
`log_final`. It moved every number in the phase:

| Measure | with the end transition | without |
| --- | ---: | ---: |
| next-word hit@1 (α = 0.1) | 17.19% | 17.20% |
| completion hit@5 (α = 0.1) | 61.99% (−0.04 vs. Phase 3) | 62.21% (+0.19) |
| windowed vs. full-sentence tag | 38.8% | 8.07% |

The hit@1 column barely moved, which is the interesting part: the headline
number would have looked fine and the defect would have shipped. It showed up
only because section 2 of the notebook prints the context tag for a worked
example, where `i would → .` is obviously nonsense. Pinned by
`test_context_tag_does_not_charge_the_end_of_sentence_transition`, which forces
the end distribution onto one tag and checks that `context_tag` ignores it.

### The ceiling: how good is the context-free candidate tag?

Against the tag the same tagger assigns *with* the sentence in front of it:

| | agreement |
| --- | ---: |
| all candidates | 92.99% |
| words seen in training | 96.77% |
| (unseen tokens in the sample) | 1,295 / 7,888 = 16.4% |

Top disagreements, in-context → context-free guess: `ADJ→NOUN` (132),
`VERB→NOUN` (127), `NOUN→VERB` (43), `NUM→NOUN` (38). The shape is exactly what
the arg-max of `P(t|w)` should produce: `NOUN` is the largest tag in English by
a wide margin, so any word whose distribution is close to balanced (`report`,
`record`, `use`) defaults towards it. Those are precisely the words a
context-sensitive decision would get right.

So the shortcut is *not* what is limiting the phase — 93% is a high ceiling.
The limit is the alphabet: twelve tags cannot say much about which of fifty
English words comes next.

A visible instance in the notebook: after `i would`, `like` is tagged `ADP`,
because across the corpus `like` really is a preposition more often than a
verb, and it gets demoted for it.

### The tagging window

`context_tag` decodes only the last few tokens. Two different losses, kept
apart because lumping them together overstates the approximation by an order of
magnitude:

| window | vs. full-buffer decode | vs. full-sentence decode |
| ---: | ---: | ---: |
| 1 | 7.86% | 11.25% |
| **2** | **0.66%** | 8.07% |
| 3 | 0.07% | 8.03% |
| 5 | 0.00% | 8.03% |
| whole buffer | 0.00% | 8.03% |

The first column is the window's own cost and it is already negligible at 2.
The second is the missing right context, which no window can fix: those words
have not been typed when the suggestion has to be drawn.

Raising `tag_context_size` past 2 also does nothing in the running app —
`predict()` has already trimmed the context to the trigram model's two words,
and the Phase 7 capture buffer holds two for the same reason. 2 is both the
setting and the ceiling; the table above only moves because the eval helper
feeds whole prefixes directly.

### Cost

| Arm | mean | p50 | p95 | p99 |
| --- | ---: | ---: | ---: | ---: |
| rerank off | 0.58 ms | 0.64 ms | 1.21 ms | 2.61 ms |
| rerank on | 0.67 ms | 0.72 ms | 1.11 ms | 2.64 ms |
| rerank on, cold caches | 0.69 ms | 0.73 ms | 1.37 ms | 2.84 ms |

Roughly 0.1 ms of a 50 ms budget, most of it the per-candidate emission
lookups rather than the trellis. Caches after 600 calls: 275 context windows,
4,351 words — small enough that cold and warm are within noise of each other,
which is itself the argument that the context-free candidate tag was the right
call. (Absolute latency on this machine varies by up to 4× with background
load; the *ratio* between arms is the number to trust.)

### Negative result: truecasing the context before tagging

The corpus is lowercased for counting, the tagger was trained on cased Treebank
text, so 20.3% of test tokens are unknown to the tagger purely from case;
running them through the Phase 1 truecase map first brings that to 17.7%. It
does not help:

| context fed to the tagger | hit@1 | hit@3 | hit@5 | tag agreement |
| --- | ---: | ---: | ---: | ---: |
| lowercased (shipped) | 17.20% | 28.98% | 35.07% | 92.99% |
| truecased | 17.14% | 28.97% | 35.04% | 92.61% |

Slightly worse on every column, so no code change. The likely reason is that
the hapax prior plus the suffix heuristics already handle a lowercased proper
noun about as well as a mis-truecased one, and the truecase map introduces its
own errors. Worth one line in the report as a hypothesis that was tested and
dropped.

### What would actually be worth doing next

In order of expected value, none of them tuning α further:

1. **A larger tagged corpus.** The tagger has ~100k tokens of Treebank sample;
   the language model has ~1M tokens of Brown. The tag prior is the weaker
   estimate by an order of magnitude.
2. **A tag trigram.** `P(tag | two preceding tags)` separates `DET ADJ ___`
   from `ADP DET ___`; the bigram cannot. That is a change to `HMMTagger`, not
   to the predictor.
3. **Contextual candidate tagging**, if it can be made cheap — worth at most
   the 7% the shortcut currently loses.

Artefacts: no new model files — Phase 6 composes `ngram_trigram_pruned.pkl` and
`hmm_english.pkl`. New config: `reranking.alpha = 0.1` (was 0.3),
`reranking.tag_context_size = 2` with the ceiling documented.

Tests: 484 passing, 1 skipped (36 new, all in `tests/test_predictor.py` except
one config key).
Deliverable notebook: `notebooks/06_pos_reranking.ipynb` (executed — worked
scoring example, α = 0 identity check, ablation, α sweep and plot, tag-guess
agreement, window table, latency).

## Phase 7 — Desktop app

Six modules under `nticipate/app/`: `win32.py` (ctypes helpers), `hooks.py`
(capture), `overlay.py` (the popup), `injector.py` (typing the completion),
`tray.py` (the tray icon and the object that wires everything together), and
`editor.py` (the fallback window). Entry point `python -m nticipate.app`, with
`--editor` and `--check`.

No new NLP. Phase 7 consumes `Predictor.suggest()` exactly as Phase 6 left it;
everything measured below is the cost of the shell around it.

### End-to-end latency

148 held-out Brown sentences replayed keystroke by keystroke through the real
pipeline (`ContextBuffer` → `Predictor.suggest` → overlay), one prediction per
keystroke — i.e. the debouncer disabled, which is the worst case it exists to
prevent.

| Metric | Value | Budget |
| --- | ---: | ---: |
| Predictions | 11,014 | — |
| p50 | **1.39 ms** | — |
| p95 | **5.40 ms** | 50 ms |
| p99 | 8.52 ms | — |
| max | 56.96 ms | — |
| mean | 1.81 ms | — |

p95 is 9× inside the debounce window, so the debouncer is saving work rather
than saving the deadline. The single 57 ms outlier is one sample in 11,014 and
is a GC pause, not a slow context — it is the only sample above 20 ms.

### The first prediction is 300× slower than the rest

| | ms |
| --- | ---: |
| Model load + app construction | 609 |
| First (cold) prediction | 363 |
| Steady-state prediction | 1.4 |

The cold call pays for NLTK's tokenizer load and the n-gram model's lazy
context index. This was found by measuring a fresh app after three keystrokes
and getting a p50 of 520 ms — an apparent 10× budget violation that was
entirely start-up cost. `NticipateApp.warmup()` now runs one throwaway
prediction before the hook starts and discards its timing, so the user's first
keystroke sees the warm path and the reported p95 describes typing rather than
launching.

Process RSS with the trigram model, the trie and the English tagger resident:
**199 MB**.

### Keystroke savings, in the app rather than in the notebook

Same replay, accepting the top suggestion whenever it matched the word being
typed (at least one character typed first, so this is the completion mode only):

| | |
| --- | ---: |
| Keystrokes typed | 11,014 |
| Keystrokes saved | 4,149 |
| Accepts | 1,904 |
| **Savings ratio** | **27.4%** |

Comparable to the Phase 3 figure, as it should be — the app changes nothing
about the ranking. The difference is that this number counts the Tab press as
a keystroke, which the pure-model measurement does not.

### Caret positioning: where it works and where it does not

`GetGUIThreadInfo` returns a caret rectangle for classic Win32 edit controls
and nothing for anything that draws its own text. Measured by launching each
app and querying the foreground window:

| Application | Focused window class | Caret | Anchor used |
| --- | --- | --- | --- |
| Notepad | `RichEditD2DPT` | **yes** — `Rect(77, 310, 78, 331)` | caret |
| VS Code (Electron) | `Chrome_RenderWidgetHostHWND` | no | mouse |
| Counter-Strike 2 (SDL) | `SDL_app` | no | mouse |

Chrome, Edge and Firefox are **not yet measured** — the browser launched for
the test never took the foreground, so the run was discarded rather than
written up. Chromium browsers use the same `Chrome_RenderWidgetHostHWND` class
as the VS Code row, so the expectation is that they behave identically, but
that is an inference and the manual check in a browser is still outstanding.

This is the phase's honest limitation. Those classes are listed in
`win32.OPAQUE_WINDOW_CLASSES` and answered with "unknown" rather than with the
stale rectangle the API happily returns for them, so the overlay falls back to
the mouse cursor instead of appearing in the wrong place with confidence.

### Focus theft

The failure that makes this class of tool unusable. Three things prevent it and
all three are needed: `overrideredirect(True)`, `WS_EX_NOACTIVATE` applied to
the top-level HWND after Tk creates it, and never binding focus to the overlay.
Verified programmatically: after `show()` the overlay is mapped
(`76x70+772+452`, mouse-anchored) and its extended style has `WS_EX_NOACTIVATE`,
`WS_EX_TOOLWINDOW` and `WS_EX_TOPMOST` all set. The by-hand check — typing in
Notepad and in a browser with the tray app running, confirming the caret never
moves — is the phase's remaining manual step.

### Accepting typed the word *and* a Tab

Found by hand-testing the tray app: Tab accepted the suggestion and then
indented the line — `word` came out as `word    `. Not an injector bug. pynput's
`on_press` is not where the key can be stopped on Windows: the low-level hook
procedure posts the event to the listener's message loop and returns, and it is
that return value which tells Windows whether to pass the key on. By the time a
callback runs, the application already has the Tab.

The one callback that runs *inside* the hook procedure is `win32_event_filter`,
so `KeystrokeHook.win32_event_filter()` is where the accept now lives:

| Constraint | Consequence in the filter |
| --- | --- |
| Only this callback can suppress | The accept has to be *routed* here, not in `on_press` |
| Hook must return before `LowLevelHooksTimeout` (~300 ms) | Callback dispatched to a worker thread |
| Injection re-enters the hook | Same — typing must not happen on the hook thread |
| A second Tab must be an ordinary Tab | `router.suggesting` cleared synchronously, before the worker runs |

It fires only for the accept key's key-down, with no modifier held, with the
overlay up and with capture unblocked; anything else returns `True` and is left
completely alone, so a plain Tab still indents and Alt+Tab still switches
windows. `app.hotkeys.suppress_accept` turns it off. The Tk editor fallback
never had the bug — it already returned `"break"` from its `<Tab>` binding,
which is the same rule expressed in the one place Tk offers it.

### Privacy

The plan's hard requirements, and how each is enforced rather than configured:

| Requirement | Enforcement |
| --- | --- |
| Nothing transmitted | No network code anywhere in `nticipate/`; `telemetry: false` |
| Buffer never on disk | `ContextBuffer` has no `save`/`to_dict`/`write` — asserted by a test |
| Buffer holds one sentence | Cleared at `.!?।॥`, capped at 200 chars |
| No leak via logs | `__repr__` reports lengths, not contents — asserted by a test |
| Password fields | `CapturePolicy` blocks capture *and clears the buffer* |

The password check is genuinely partial and is reported as such: `ES_PASSWORD`
is readable on Win32 edit controls, and browsers expose nothing, so an unknown
falls back to a window-title heuristic (`sign in`, `password`, …). A browser
password box in a page with an uninformative title will not be detected.

Personalization is the one feature that writes what the user typed to disk, so
it is **off by default** (`app.learning.enabled: false`) and turned on from the
tray menu. What is saved is the profile's n-gram counts, never the keystroke
stream.

Windows Defender may flag the global hook: `pynput.keyboard.Listener` installs
a low-level keyboard hook, which is structurally what a keylogger installs.
Develop inside an exclusion folder and disclose it in the write-up — the
mitigation is disclosure, not evasion.

### The fallback was built anyway

`python -m nticipate.app --editor` opens a Tk editor with live suggestions,
using the same predictor and the same accept arithmetic, without `pynput`,
without a hook and without caret guessing. It was written even though the hook
works, because it is the demo that cannot fail in front of an examiner. Same
measured latency (mean 1.5 ms over a session), and it shows each suggestion's
POS tag, which the overlay does not.

Artefacts: no new model files. New config keys: `app.models.*`,
`app.capture.{append_space, max_buffer_chars, sentence_end_chars}`,
`app.learning.{enabled, autosave_every}`, `app.hotkeys.suppress_accept`.

Tests: 609 passing, 1 skipped (113 new, `tests/test_app.py` plus config-key
assertions in `test_phase0_setup.py`). No notebook — Phase 7's deliverable is
the running app, and the numbers above come from a replay harness rather than
from a per-phase notebook.

## Phase 8 — Packaging

_(exe size, RAM, startup time)_
