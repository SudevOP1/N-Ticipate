# Lab notebooks

One notebook per lab experiment, numbered by phase. Each should be
self-contained: load config, run the experiment, produce the plot/table
that goes straight into `report/`.

- `01_preprocessing.ipynb` — cleaning, tokenization, vocab/UNK, Zipf plot
- `02_ngram_models.ipynb` — bigram/trigram counts, smoothing sweep, perplexity
- `03_prediction_engine.ipynb` — latency + hit@k benchmarks
- `04_hmm_english.ipynb` — Viterbi trellis worked example, accuracy, confusion matrix
- `05_hmm_regional.ipynb` — Hindi/Marathi tagging, TTR/OOV comparison vs. English
- `06_pos_reranking.ipynb` — ablation: hit@k with and without reranking
