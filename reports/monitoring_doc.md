# Monitoring & Retraining Strategy

**Healthcare RAG-Powered Medical Q&A Assistant**
**Owner:** Ziad Ahmed El-Nady | eyouth x DEPI 2025

---

## 1. Metrics Monitored

| Metric | Tool | Frequency |
|--------|------|-----------|
| ROUGE-L score | MLflow + evaluation notebook | Per experiment run |
| Macro F1 (classifier) | MLflow | Per training run |
| API response latency (ms) | FastAPI middleware + logs | Per request |
| Query volume by category | Streamlit dashboard | Real-time |
| Hallucination rate | Manual review | Every 30 responses |

---

## 2. Drift Thresholds

| Metric | Healthy | Warning | Trigger Retraining |
|--------|---------|---------|-------------------|
| ROUGE-L | >= 0.15 | 0.10 - 0.15 | < 0.10 over 7-day window |
| Macro F1 | >= 0.80 | 0.72 - 0.80 | < 0.72 on new query sample |
| Avg latency | <= 3,500 ms | 3,500 - 5,000 ms | > 5,000 ms sustained warm |
| Hallucination rate | <= 10% | 10 - 15% | > 15% on manual review |

Drift is measured as a rolling 7-day average against the baselines recorded
in MLflow experiment run `baseline_topk5`.

---

## 3. Retraining Steps

### 3a. RAG pipeline retraining
1. Download updated PubMedQA data (rerun notebook 01)
2. Re-run preprocessing pipeline (notebook 02)
3. Re-label categories (notebook 03)
4. Regenerate embeddings and rebuild FAISS index (notebook 05)
5. Re-evaluate BLEU/ROUGE-L (notebook 08)
6. Log new experiment run: `python mlops/mlflow_tracking.py`
7. If metrics improve, re-upload FAISS index to HuggingFace dataset repo

### 3b. Classifier retraining
1. Re-run fine-tuning on updated labelled data (notebook 07)
2. Evaluate on held-out test set — target macro F1 >= 0.78
3. If improved, upload new weights: `python scripts/upload_classifier_to_hub.py`
4. Register new model version in MLflow registry

---

## 4. Infrastructure Constraints

This monitoring strategy is documented as a **future enhancement**.
Current deployment uses Azure App Service Free Tier (F1), which does not support:
- Continuous monitoring agents
- Automated retraining pipelines
- Scheduled jobs or cron tasks

Automated drift detection and retraining would require upgrading to at minimum
Azure App Service Basic tier (B1) and adding Azure Monitor alerts.

---

## 5. Current Baseline (from MLflow run: baseline_topk5)

| Metric | Baseline Value |
|--------|---------------|
| BLEU (RAG) | 0.0157 |
| ROUGE-L (RAG) | 0.1663 |
| BLEU improvement over LLM | 1862.5% |
| Classifier Macro F1 | 0.867 |
| Avg latency (warm) | 3,197 ms |
| Hallucination rate | 10% |

*All baselines recorded May 2026. Retraining triggered when any metric
degrades beyond the thresholds in Section 2.*
