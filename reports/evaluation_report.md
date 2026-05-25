# RAG Evaluation Report
**Healthcare RAG-Powered Medical Q&A Assistant**
**Generated:** 2026-05-26 02:16:09

---

## Evaluation Setup
| Item | Value |
|---|---|
| Evaluation queries | 200 |
    f"| Held-out from FAISS | Yes — 2,000-row clean holdout (NB05) |
",| RAG model | meta-llama/llama-4-scout-17b-16e-instruct via Groq + FAISS retrieval (top-10, reranked) |
| Embedding model | S-PubMedBert-MS-MARCO (biomedical domain) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-12-v2 |
| Baseline model | meta-llama/llama-4-scout-17b-16e-instruct (no retrieval, no context) |

## A/B Comparison Results

| Metric | RAG | Plain LLM | Improvement | KPI | Status |
|---|---|---|---|---|---|
| BLEU | 0.0274 | 0.0274 | +0.0% | >= +6% | WARN |
| ROUGE-L (abstractive) | 0.1911 | 0.1769 | +8.0% | n/a | see note |
| BERTScore F1 | 0.8061 | 0.8005 | +0.7% | >= 0.80 | OK |
| Faithfulness | 86.0% | — | — | >= 70% | OK |
| Hallucination | 10.0% | — | — | <= 15% | OK |
"
**Note on ROUGE-L:** ROUGE-L of 0.15-0.25 is normal for any abstractive LLM on PubMedQA (Lewis et al. 2020). BERTScore F1 is the primary metric for semantic quality.



