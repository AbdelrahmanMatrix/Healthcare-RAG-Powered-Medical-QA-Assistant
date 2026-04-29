# RAG Evaluation Report
**Healthcare RAG-Powered Medical Q&A Assistant**
**Owner:** Eman Khalid Ismail
**Generated:** 2026-04-29 14:53:44

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
| BLEU | 0.1876 | 0.0008 | 23350.0% |
| ROUGE-L | 0.2897 | 0.0263 | 1001.5% |

## KPI Status

| KPI | Target | Actual | Status |
|-----|--------|--------|--------|
| ROUGE-L | ≥ 0.38 | 0.2897 | ⚠️ NOT MET |
| BLEU improvement | ≥ 20% | 23350.0% | ✅ MET |
| Hallucination rate | ≤ 15% | 10.0% | ✅ MET |

## Hallucination Review
- Samples reviewed: 30
- Hallucinated responses: 3
- Hallucination rate: 10.0%

## RAG Latency
- Mean: 1458ms
- Min: 549ms
- Max: 7295ms

**Status: M2 Task 4 — Completed**
