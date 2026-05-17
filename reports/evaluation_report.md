# RAG Evaluation Report
**Healthcare RAG-Powered Medical Q&A Assistant**
**Owner:** Eman Khalid Ismail
**Generated:** 2026-05-17 17:56:26

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
| BLEU | 0.0157 | 0.0008 | 1862.5% |
| ROUGE-L | 0.1663 | 0.0263 | 532.3% |

## KPI Status

| KPI | Target | Actual | Status |
|-----|--------|--------|--------|
| ROUGE-L | ≥ 0.38 | 0.1663 | ⚠️ NOT MET |
| BLEU improvement | ≥ 20% | 1862.5% | ✅ MET |
| Hallucination rate | ≤ 15% | 10.0% | ✅ MET |

## Hallucination Review
- Samples reviewed: 30
- Hallucinated responses: 3
- Hallucination rate: 10.0%

## RAG Latency
- Mean: 3197ms
- Min: 1569ms
- Max: 14886ms

**Status: M2 Task 4 — Completed**
