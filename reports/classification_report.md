# Classification Report — DistilBERT Medical Classifier

## Model Details
| Item | Value |
|---|---|
| Base model | `distilbert-base-uncased` |
| Classes | 6 |
| Training split | 80/10/10 |
| Epochs | 3 |
| Learning rate | 2e-5 |
| Batch size | 16 |
| Class weights | Applied (balanced) |

## Test Set Results
              precision    recall  f1-score   support

   Diagnosis       0.86      0.88      0.87       184
     General       0.90      0.90      0.90       135
  Medication       0.81      0.83      0.82       149
  Prevention       0.81      0.88      0.84       115
    Symptoms       0.84      0.85      0.84        54
   Treatment       0.92      0.86      0.89       363

    accuracy                           0.87      1000
   macro avg       0.85      0.87      0.86      1000
weighted avg       0.87      0.87      0.87      1000


## Key Metrics
| Metric | Value |
|---|---|
| Macro F1 | 0.8597 |
| Weighted F1 | 0.8675 |
| Accuracy | 0.8670 |
| KPI (Macro F1 ≥ 0.78) | ✅ MET |

## Label Mapping
| Diagnosis | 0 |
| General | 1 |
| Medication | 2 |
| Prevention | 3 |
| Symptoms | 4 |
| Treatment | 5 |
