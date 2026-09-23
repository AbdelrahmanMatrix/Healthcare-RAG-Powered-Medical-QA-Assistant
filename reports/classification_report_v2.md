# Classification Report v2 — BioBERT Medical Classifier (post-leakage-repair)

> **Status: SPLIT VERIFIED, METRICS PENDING RETRAIN.** The data split below was
> reproduced and verified locally (`reports/p04_split_verification.json`; the
> reproduced holdout matches the published `data/processed/eval_holdout.csv`
> artifact exactly, including row order). The metrics table is filled in by
> notebook 07 after the retrain completes. Until then, the numbers in
> `reports/classification_report.md` are HISTORICAL and must not be quoted as
> current performance.

## Label provenance — weak labels
All six categories are **programmatic keyword labels** produced by
`src/data/labeller.py` (notebook 03). They are **not human annotations**:
per-class metrics measure agreement with the keyword rules, and systematic
labeller errors are invisible to these metrics.

## Data split (verified, deterministic)
| Item | Value |
|---|---|
| Labelled rows loaded (`pubmedqa_labelled.csv`) | 211,186 |
| After dropna/strip cleaning (NB05 logic) | 211,186 |
| Duplicates removed (same question) | 78 |
| Rows after dedup | 211,108 |
| RAG holdout excluded (never trained on) | 2,000 |
| Train / Val / Test (stratified, random_state=42) | 167,286 / 20,911 / 20,911 |
| Holdout overlap with any classifier split | 0 rows |

Input text: `question + " [SEP] " + context` (BioBERT is cased — original
capitalisation preserved).

## Model & training configuration
| Item | Value |
|---|---|
| Base model | `dmis-lab/biobert-v1.1` |
| Max sequence length | 256 |
| Epochs | 10 configured, early stopping (patience=2, metric: val f1_macro) |
| Learning rate / batch | 2e-5 / 16 |
| Weight decay / warmup | 0.01 / 500 |
| Class weights | Balanced, recomputed on the post-exclusion train split |
| Seed | 42 |

## Metrics (test split, n = 20,911)
| Metric | Value |
|---|---|
| Macro F1 | *PENDING RETRAIN* |
| Weighted F1 | *PENDING RETRAIN* |
| Accuracy | *PENDING RETRAIN* |
| Best epoch (by val f1_macro) | *PENDING RETRAIN* |

Per-class report, confusion matrix, and KPI check are emitted by notebook 07
into this file on completion, alongside `reports/classifier_training_manifest.json`
(full environment/library versions).
