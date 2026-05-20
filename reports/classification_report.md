# Classification Report — BioBERT Medical Classifier

## Model Details
| Item | Value |
|---|---|
| Base model | `dmis-lab/biobert-v1.1` |
| Classes | 6 |
| Training split | 80/10/10 |
| Epochs | 3 |
| Learning rate | 2e-5 |
| Batch size | 16 |
| Class weights | Applied (balanced) |

## Test Set Results
              precision    recall  f1-score   support

   Diagnosis       0.90      0.91      0.91      3192
     General       0.80      0.97      0.88      2825
  Medication       0.96      0.88      0.92      7154
  Prevention       0.92      0.91      0.91      2218
    Symptoms       0.93      0.89      0.91       872
   Treatment       0.91      0.91      0.91      4858

    accuracy                           0.91     21119
   macro avg       0.90      0.91      0.91     21119
weighted avg       0.91      0.91      0.91     21119


## Key Metrics
| Metric | Value |
|---|---|
| Macro F1 | 0.9066 |
| Weighted F1 | 0.9094 |
| Accuracy | 0.9088 |
| KPI (Macro F1 ≥ 0.78) | ✅ MET |

## Label Mapping
| Diagnosis | 0 |
| General | 1 |
| Medication | 2 |
| Prevention | 3 |
| Symptoms | 4 |
| Treatment | 5 |
