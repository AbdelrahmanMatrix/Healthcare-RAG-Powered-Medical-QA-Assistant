# RAG Evaluation Report
**Healthcare RAG-Powered Medical Q&A Assistant**
**Owner:** Eman Khalid Ismail
**Generated:** 2026-05-06 15:24:02

---

## Evaluation Setup
| Item | Value |
|---|---|
| Evaluation queries | 200 |
| Held-out from FAISS | Partial (tail split) |
| RAG model | flan-t5-base + FAISS retrieval (top-5) |
| Baseline model | flan-t5-base (no retrieval) |
| Embedding model | all-MiniLM-L6-v2 |

## A/B Comparison Results

| Metric | RAG | Plain LLM | Improvement |
|--------|-----|-----------|-------------|
| BLEU | 0.0198 | 0.0008 | 2375.0% |
| ROUGE-L | 0.1563 | 0.0263 | 494.3% |

## KPI Status

| KPI | Target | Actual | Status |
|-----|--------|--------|--------|
| ROUGE-L | ≥ 0.38 | 0.1563 | ⚠️ NOT MET |
| BLEU improvement | ≥ 20% | 2375.0% | ✅ MET |
| Hallucination rate | ≤ 15% | 10.0% | ✅ MET |

## Hallucination Review
- Samples reviewed: 30
- Hallucinated responses: 3
- Hallucination rate: 10.0%

## RAG Latency
- Mean: 5993ms
- Min: 3268ms
- Max: 35111ms

**Status: M2 Task 4 — Completed**
