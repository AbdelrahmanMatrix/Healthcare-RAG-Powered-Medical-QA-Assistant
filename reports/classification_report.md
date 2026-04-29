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

   Diagnosis       0.70      0.96      0.81       305
     General       0.62      0.62      0.62        29
  Medication       0.56      0.76      0.65        29
  Prevention       0.38      0.60      0.46        15
    Symptoms       0.97      0.74      0.84       589
   Treatment       0.68      0.76      0.71        33

    accuracy                           0.80      1000
   macro avg       0.65      0.74      0.68      1000
weighted avg       0.85      0.80      0.81      1000


## Key Metrics
| Metric | Value |
|---|---|
| Macro F1 | 0.6821 |
| Weighted F1 | 0.8086 |
| Accuracy | 0.8040 |
| KPI (Macro F1 ≥ 0.78) | ⚠️ NOT MET |

## Label Mapping
| Diagnosis | 0 |
| General | 1 |
| Medication | 2 |
| Prevention | 3 |
| Symptoms | 4 |
| Treatment | 5 |
