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

   Diagnosis       0.72      0.96      0.82       305
     General       0.65      0.76      0.70        29
  Medication       0.60      0.72      0.66        29
  Prevention       0.50      0.73      0.59        15
    Symptoms       0.97      0.77      0.86       589
   Treatment       0.66      0.76      0.70        33

    accuracy                           0.82      1000
   macro avg       0.68      0.78      0.72      1000
weighted avg       0.86      0.82      0.83      1000


## Key Metrics
| Metric | Value |
|---|---|
| Macro F1 | 0.7228 |
| Weighted F1 | 0.8291 |
| Accuracy | 0.8250 |
| KPI (Macro F1 ≥ 0.78) | ⚠️ NOT MET |

## Label Mapping
| Diagnosis | 0 |
| General | 1 |
| Medication | 2 |
| Prevention | 3 |
| Symptoms | 4 |
| Treatment | 5 |
