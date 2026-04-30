# RAG Evaluation Report
**Healthcare RAG-Powered Medical Q&A Assistant**
**Owner:** Eman Khalid Ismail
**Generated:** 2026-04-30 17:45:06

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
| BLEU | 0.0053 | 0.0008 | 562.5% |
| ROUGE-L | 0.0670 | 0.0263 | 154.8% |

## KPI Status

| KPI | Target | Actual | Status |
|-----|--------|--------|--------|
| ROUGE-L | ≥ 0.38 | 0.0670 | ⚠️ NOT MET |
| BLEU improvement | ≥ 20% | 562.5% | ✅ MET |
| Hallucination rate | ≤ 15% | 10.0% | ✅ MET |

## Hallucination Review
- Samples reviewed: 30
- Hallucinated responses: 3
- Hallucination rate: 10.0%

## RAG Latency
- Mean: 595ms
- Min: 268ms
- Max: 1421ms

**Status: M2 Task 4 — Completed**
