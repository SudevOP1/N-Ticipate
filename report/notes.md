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

_(accuracy side-by-side with English, OOV rate, confused tag pairs)_

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

_(apps tested, caret-positioning successes and failures)_

## Phase 8 — Packaging

_(exe size, RAM, startup time)_
