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

   Diagnosis       0.87      0.86      0.86       184
     General       0.88      0.90      0.89       135
  Medication       0.86      0.89      0.87       149
  Prevention       0.79      0.90      0.84       115
    Symptoms       0.82      0.83      0.83        54
   Treatment       0.93      0.88      0.90       363

    accuracy                           0.88      1000
   macro avg       0.86      0.87      0.87      1000
weighted avg       0.88      0.88      0.88      1000


## Key Metrics
| Metric | Value |
|---|---|
| Macro F1 | 0.8660 |
| Weighted F1 | 0.8786 |
| Accuracy | 0.8780 |
| KPI (Macro F1 ≥ 0.78) | ✅ MET |

## Label Mapping
| Diagnosis | 0 |
| General | 1 |
| Medication | 2 |
| Prevention | 3 |
| Symptoms | 4 |
| Treatment | 5 |
