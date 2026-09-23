# Retrieval-Only Diagnostics (Final Validation)

**Eval sample:** 200 queries, seed 42, from `data/processed/eval_holdout.csv` (2,000-row RAG holdout)
**Index rows:** 209,108 | **top_k:** 30 | **BM25 threshold:** 12.0 | **expansion:** {'Medication': 2, 'Treatment': 2, 'Diagnosis': 3, 'General': 3, 'Prevention': 4, 'Symptoms': 5}
**Leak check:** holdout chunk_ids present in index range: not verifiable at id level (no chunk_id column in CSV)
**Measured queries:** 200 of 200 (guard-skipped: 0, classifier errors: 0)

## A. Top-k / fusion behavior
- Unique candidates after fusion (before truncate-to-30): mean 104.5, median 109.0, range 69-170
- Queries where the fusion pool exceeded top_k (truncation applied): 200/200

## B. Retrieval contributions
- BM25 threshold survivors per query: mean 29.76, median 30.0, max 30
- Queries with at least 1 BM25 survivor entering the pool: 200/200
- BM25 added a candidate absent from the FULL expanded FAISS pool: 200/200 (100.0%)
- BM25 added a candidate absent from FAISS top-30: 200/200 (100.0%)
- FAISS contributed candidates absent from BM25 survivors: 200/200 (100.0%)
- Cross-encoder rerank latency per predict call (30 pairs): mean 1497.6 ms, p95 2103.8 ms (CPU)

## C. Reranker / guard behavior
- Reranker changed the top result vs the category-priority pool order: 168/200 (84.0%)
- Final top-1 differs from FAISS-only top-1: 91/200 (45.5%)
- Final top-1 agrees with FAISS-only top-1: 54.5%
- Category quality-guard fallback (top-3 mean reranker score < 1.0) would trigger: 67/200 (33.5%)

## D. Classifier routing (retrieval context)
- Predicted categories: {'Medication': 32, 'General': 114, 'Symptoms': 5, 'Treatment': 31, 'Prevention': 7, 'Diagnosis': 11}
- Confidence: mean 0.8926, min 0.3985
- Guard-skipped (no-retrieval) queries: 0

**Status:** measurements only. No optimization performed. Generation-dependent metrics NOT run here (canonical LLM evaluation is blocked on this machine: no GROQ_API_KEY).