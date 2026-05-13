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

   Diagnosis       0.88      0.85      0.87       184
     General       0.87      0.97      0.92       135
  Medication       0.84      0.83      0.84       149
  Prevention       0.85      0.86      0.86       115
    Symptoms       0.79      0.81      0.80        54
   Treatment       0.91      0.88      0.90       363

    accuracy                           0.88      1000
   macro avg       0.86      0.87      0.86      1000
weighted avg       0.88      0.88      0.88      1000


## Key Metrics
| Metric | Value |
|---|---|
| Macro F1 | 0.8628 |
| Weighted F1 | 0.8758 |
| Accuracy | 0.8760 |
| KPI (Macro F1 ≥ 0.78) | ✅ MET |

## Label Mapping
| Diagnosis | 0 |
| General | 1 |
| Medication | 2 |
| Prevention | 3 |
| Symptoms | 4 |
| Treatment | 5 |
