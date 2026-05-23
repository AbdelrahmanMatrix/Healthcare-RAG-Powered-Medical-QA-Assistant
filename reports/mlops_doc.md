# MLOps Setup Summary

**Healthcare RAG-Powered Medical Q&A Assistant**
**Owner:** Youssef George Youssef | eyouth x DEPI 2025

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

### 5 experiment configurations run

| Run name | top_k | inject_k | max_context_words |
|----------|-------|----------|------------------|
| baseline_topk5 | 5 | 3 | 200 |
| topk3_tighter | 3 | 3 | 200 |
| topk5_more_context | 5 | 5 | 300 |
| topk7_wide | 7 | 3 | 200 |
| inject5_wide_context | 7 | 5 | 350 |

---

## 2. MLflow Model Registry

**Registered model name:** `healthcare-rag-classifier`
**Production version:** v1 (run: `topk3_tighter` — lowest latency at 1,040 ms)
**Selection criteria:** lowest `avg_latency_ms` among all 5 runs

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
