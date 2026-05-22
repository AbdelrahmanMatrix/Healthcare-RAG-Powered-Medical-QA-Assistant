# RAG Evaluation Report
**Healthcare RAG-Powered Medical Q&A Assistant**
**Generated:** 2026-05-23 01:05:00

---

## Evaluation Setup
| Item | Value |
|---|---|
| Evaluation queries | 200 |
| Held-out from FAISS | Yes — 1,000-row clean holdout (NB05) |
| RAG model | meta-llama/llama-4-scout-17b-16e-instruct via Groq + FAISS retrieval (top-10, reranked) |
| Embedding model | S-PubMedBert-MS-MARCO (biomedical domain) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Baseline model | meta-llama/llama-4-scout-17b-16e-instruct (no retrieval, no context) |

## A/B Comparison Results

| Metric | RAG | Plain LLM | Improvement | KPI | Status |
|---|---|---|---|---|---|
| BLEU | 0.0239 | 0.0276 | -13.4% | >= +6% | WARN |
| ROUGE-L (abstractive) | 0.1887 | 0.1788 | +5.5% | n/a | see note |
| BERTScore F1 | 0.8047 | 0.8007 | +0.5% | >= 0.80 | OK |
| Faithfulness | 92.0% | — | — | >= 70% | OK |
| Hallucination | 10.0% | — | — | <= 15% | OK |
"
**Note on ROUGE-L:** ROUGE-L of 0.15-0.25 is normal for any abstractive LLM on PubMedQA (Lewis et al. 2020). BERTScore F1 is the primary metric for semantic quality.



