---
language: en
license: mit
tags:
  - medical
  - classification
  - distilbert
  - pubmedqa
  - healthcare-rag
datasets:
  - llamafactory/PubMedQA
metrics:
  - f1
pipeline_tag: text-classification
---

# DistilBERT Medical Query Classifier

Fine-tuned distilbert-base-uncased for classifying medical questions into 6 categories.

## Categories
| ID | Category |
|----|----------|
| 0 | Diagnosis |
| 1 | General |
| 2 | Medication |
| 3 | Prevention |
| 4 | Symptoms |
| 5 | Treatment |

## Results
| Metric | Score |
|--------|-------|
| Macro F1 | 0.8491 |
| Weighted F1 | 0.8588 |
| Accuracy | 0.8580 |

## Training Config
| Item | Value |
|------|-------|
| Base model | distilbert-base-uncased |
| Dataset | llamafactory/PubMedQA (10,000 rows) |
| Split | 80/10/10 |
| Epochs | 3 |
| Learning rate | 2e-5 |
| Batch size | 16 |
| Class weights | Balanced (custom WeightedTrainer) |

## Usage
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

tokenizer = DistilBertTokenizer.from_pretrained("AbdoMatrix/distilbert-medical-classifier")
model = DistilBertForSequenceClassification.from_pretrained("AbdoMatrix/distilbert-medical-classifier")

text = "What are the symptoms of diabetes?"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)

with torch.no_grad():
    outputs = model(**inputs)

predicted = model.config.id2label[torch.argmax(outputs.logits, dim=1).item()]
print(predicted)  # → Symptoms
## Project
Healthcare RAG-Powered Medical Q&A Assistant
eyouth x DEPI | Microsoft Machine Learning Track | 2026
GitHub: https://github.com/AbdooMatrix/Healthcare-RAG-Powered-Medical-QA-Assistant
