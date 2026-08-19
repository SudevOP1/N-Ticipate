# Notebooks

One notebook per phase deliverable. Each is a presentable artefact for the lab
report: the module under `nticipate/` holds the implementation, the notebook
shows the analysis, tables and plots.

| Notebook                     | Phase | Deliverable                                     |
| ---------------------------- | ----- | ----------------------------------------------- |
| `01_preprocessing.ipynb`     | 1     | Token counts, Zipf plot, TTR, vocab coverage    |
| `02_ngram_models.ipynb`      | 2     | Perplexity table, generated samples, model size |
| `03_prediction_engine.ipynb` | 3     | hit@k, keystroke savings, latency               |
| `04_hmm_english.ipynb`       | 4     | Tagger accuracy, confusion matrix, trellis      |
| `05_hmm_regional.ipynb`      | 5     | Hindi vs. English accuracy, OOV analysis        |
| `06_pos_reranking.ipynb`     | 6     | The ablation: hit@k with vs. without reranking  |

Run from the project root so `import nticipate` resolves.
