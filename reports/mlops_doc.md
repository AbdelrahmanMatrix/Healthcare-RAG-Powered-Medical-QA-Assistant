# MLOps Setup Summary

**Healthcare RAG-Powered Medical Q&A Assistant**
**Owner:** Youssef George Youssef | eyouth x DEPI 2026

---

## 1. MLflow Experiment Tracking

**Tracking URI:** local `mlruns/` directory (file-based)
**Experiment name:** `healthcare-rag-experiments`
**Script:** `mlops/mlflow_tracking.py`

### Logged parameters (per run)
- `embedding_model` — sentence-transformers model name
- `top_k` — number of FAISS chunks retrieved
- `inject_k` — number of chunks injected into LLM prompt
- `max_context_words` — per-chunk context truncation limit
- `llm_model` — generation model identifier

### Logged metrics (per run)
- `bleu_rag` / `bleu_baseline` — BLEU scores for RAG vs plain LLM
- `rouge_rag` / `rouge_baseline` — ROUGE-L scores
- `bleu_improvement_pct` — relative improvement percentage
- `faiss_index_size` — number of vectors in the FAISS index
- `avg_latency_ms` — estimated response latency

### 6 experiment configurations run

| Run name | top_k | inject_k | max_context_words |
|----------|-------|----------|------------------|
| pubmedbert_topk10_inject3 | 10 | 3 | 200 |
| pubmedbert_topk5_inject3 | 5 | 3 | 200 |
| pubmedbert_topk10_inject5 | 10 | 5 | 350 |
| baseline_miniLM_topk5 | 5 | 3 | 200 |
| baseline_miniLM_topk10 | 10 | 3 | 200 |
| pubmedbert_topk10_inject5_ctx350 | 10 | 5 | 350 |

---

## 2. MLflow Model Registry

**Registered model name:** `healthcare-rag-classifier`
**Production version:** v1 (run: `pubmedbert_topk5_inject3` — lowest top_k among PubMedBERT configurations)
**Selection criteria:** highest `bleu_rag` among all 6 runs (latency also logged for reference)

Start the MLflow UI to inspect all runs:
```
mlflow ui
# Open: http://localhost:5000
```

---

## 3. Streamlit KPI Dashboard

**Script:** `dashboard/app.py`
**Launch:** `streamlit run dashboard/app.py`
**URL:** http://localhost:8501

### 4 dashboard sections

| Section | Data source | KPI shown |
|---------|-------------|-----------|
| System Overview | pubmedqa_labelled.csv + rag_pipeline_test_log.json | Corpus size, test queries run |
| Model Performance | rag_evaluation_results.csv + MLflow runs | BLEU/ROUGE-L trends |
| Query Analytics | pubmedqa_labelled.csv + rag_pipeline_test_log.json | Category distribution, latency |
| Data Quality | pubmedqa_cleaned.csv + pubmedqa_labelled.csv | Null counts, category coverage |

---

## 4. End-to-End System Test

**Notebook:** `notebooks/10_end_to_end_test.ipynb`
**Output:** `reports/rag_pipeline_test_log.json` (10 queries across all 6 categories)

The test verified the full pipeline: query -> classify -> retrieve -> generate -> disclaimer.
All 10 queries returned valid responses with `disclaimer_present: true`.

---

## 5. Retraining Strategy

See `reports/monitoring_doc.md` for full drift thresholds and retraining steps.
