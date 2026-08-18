# Design decisions worth writing up

Running log of choices that need justifying in the final report.

- **Stopwords and punctuation are kept, not stripped.** Standard for
  classification, wrong for language modeling — "of the" is exactly what
  a bigram model needs to predict. (Phase 1)
- **Stupid backoff ships in the app; Kneser-Ney is the reported ceiling.**
  Backoff is O(1) lookup with no renormalization, which matters at
  keystroke latency. KN is more accurate but heavier — good for the
  perplexity comparison, not for the live predictor. (Phase 2)
- **HMM tagger trained from scratch**, NLTK's trainer used only as a
  correctness cross-check. (Phase 4)
- **Counting is case-insensitive, display isn't.** `apply_unk` lowercases
  before vocab lookup (`CFG['preprocessing']['lowercase_for_counts']`),
  so "The" and "the" share one n-gram context instead of splitting the
  counts. A separate truecase map (`build_truecase_map` /
  `apply_truecase`) restores the majority-seen surface casing at
  suggestion time — without it the app would suggest "india" mid-sentence.
  (Phase 1)
- **Tokenizer prefers NLTK's TreebankWordTokenizer** (correctly splits
  "don't" -> "do" + "n't"), with a regex fallback so the module still
  works before `setup_env.py` downloads Punkt data. The fallback does
  *not* split contractions — noted so it isn't mistaken for the real
  behaviour when comparing dev output against the shipped app. (Phase 1)
- **Kneser-Ney's unigram level uses continuation probability**, not raw
  unigram frequency (Chen & Goodman) — how many *distinct* words precede
  it, not how often it occurs. Needs the bigram model's counts, so
  `build_ngram_hierarchy` installs it after both orders are built rather
  than each `NgramModel` computing it standalone. (Phase 2)
- **Stupid backoff ships in the app**: on the sample corpus it perplexity-
  ties with Kneser-Ney at n=2/n=3 (≈7-8 vs ≈7) while being O(1) lookup
  with no renormalization step — worth the small quality gap at keystroke
  latency. Re-check this gap on the full NLTK corpora; it may widen.
  (Phase 2)
- **Pruning trades quality for size on purpose.** On the sample corpus,
  pruning cut the trigram model from 5.9KB/198 contexts to 0.9KB/32
  contexts while perplexity rose from 18.1 to 22.4 — that gap is the
  size/quality trade-off the packaged app is making, not a bug. (Phase 2)
- **Candidate pool must include the user model's own vocabulary, not
  just the base model's.** First implementation only pulled prediction
  candidates from the base n-gram model, so a word the user typed
  hundreds of times (e.g. their own name) could never surface no matter
  how high `lambda` climbed — it was simply never in the dict being
  scored. Fixed by unioning `ngram_model.top_k()` candidates with
  `user_profile.top_k()` candidates (next-word) or `user_profile.vocab`
  matches (prefix completion) *before* scoring. Caught by actually
  running personalization end-to-end with a synthetic user, not by
  reading the code. (Phase 3)
- **Personalization doesn't UNK the user's own words.** `UserProfile`
  skips `apply_unk`/`build_vocab` entirely — the whole point is
  capturing exactly the words the base corpus would otherwise discard
  (names, jargon). Its own small n-gram hierarchy is refit lazily
  (`_dirty` flag) rather than on every keystroke. (Phase 3)
- **HMM keeps original casing, never routes through preprocess.py's
  lowercasing.** Capitalization is itself a POS signal (proper nouns) —
  lowercasing before tagging would throw away exactly the feature the
  unknown-word suffix heuristic relies on. (Phase 4)
- **Unknown-word emissions use a 50/50 split**, not a proper statistical
  model (e.g. Good-Turing): half the probability mass goes to the
  suffix-guessed tag, the rest is split across every other tag. Simple,
  and it worked — test accuracy landed at 78% against a 25% most-common-
  tag baseline on the sample corpus — but it's a real simplification
  worth naming explicitly rather than dressing up as more rigorous than
  it is. (Phase 4)
- **Train accuracy (83.5%) is below 100% on purpose, not a bug.** Words
  like "her" are genuinely ambiguous (DET in "her laptop" vs. PRON in
  "helps her feel") and the emission table can't distinguish which
  occurrence it's looking at — even on training data, Viterbi can pick
  the globally more likely tag and get a specific instance wrong. That
  gap is evidence of real statistical inference, not memorization; report
  it as a finding, not an error. (Phase 4)
- **Same `HMMTagger` class trained on Hindi with zero code changes** —
  only the corpus and the (already-present) Devanagari branch of
  `_guess_tag_by_suffix` differ. Test accuracy: 78.0% (English) vs.
  67.4% (Hindi) on comparably-sized hand-tagged samples, and the gap is
  consistent with the TTR/OOV numbers (Hindi TTR 0.667 vs. English
  0.518; OOV rate 0.838 vs. 0.770) — the morphological-richness
  hypothesis from the plan, backed by matching numbers across three
  independent metrics rather than asserted on its own. (Phase 5)
- **Didn't chase a better suffix-heuristic number by adding more rules.**
  OOV accuracy on Hindi test words came out at 47.8% — most misses were
  verb forms ending in `ा`/`ी`/`ीं` that the rule list doesn't cover.
  Adding those endings would fix some cases, but `ी` is *also* the
  standard feminine noun marker (लड़की = "girl"), so the same rule
  would break noun classification elsewhere. This is genuine
  morphological overlap, not a missing-rule gap — tuning against a
  23-word OOV test set would just be overfitting to sample noise. Named
  as a limitation with its actual cause, rather than patched to look
  better. (Phase 5)
- **No real nltk `indian` corpus in this sandbox (no network access)** —
  used 35 original hand-tagged Hindi sentences instead, deliberately
  using the *same* Universal tagset as Phase 4's English corpus so the
  two are directly comparable. The real `indian` corpus uses its own
  native (IIIT-Hyderabad) tagset; no mapping to Universal was attempted
  since an unverifiable mapping risked being simply wrong — the
  `CORPUS_SOURCE = "nltk"` branch trains on native tags as-is instead.
  (Phase 5)
- **Reranking nudges, it doesn't override.** `alpha = 0.3` means a 5×
  base-probability gap survives reranking untouched — only genuinely
  close calls flip toward the POS-plausible candidate. Caught this by
  writing a test with an unrealistic 5× gap that (correctly) didn't
  flip, then realizing the test's expectation was wrong, not the code.
  (Phase 6)
- **Candidate word tags are guessed context-free** via
  `most_likely_tag_for_word` (argmax emission probability, or the
  suffix heuristic for OOV words) — necessarily coarser than
  `viterbi()`'s context-aware tagging, since a candidate is just a
  vocabulary string with no sentence around it yet. Also handles the
  n-gram/HMM vocabulary-casing mismatch (n-gram vocab is lowercased,
  HMM vocab keeps original casing) by trying the word as given, then
  title-cased, before falling back to the suffix guess. (Phase 6)
- **The hit@k ablation came out mixed on the sample corpus** (hit@1
  slightly down, hit@3 flat, hit@5 slightly up) — reported as-is rather
  than as a clean win. At this sample size there isn't enough held-out
  data to distinguish a real effect from noise; the honest conclusion
  is "re-run on the full corpus before drawing a conclusion," not "the
  reranker doesn't work." The qualitative disagreement cases (section 4
  of the notebook) show it's clearly doing something sensible — pushing
  non-noun candidates below noun candidates after a determiner — even
  where it doesn't happen to hit the exact held-out word. (Phase 6)
- *(add more as each phase lands)*
