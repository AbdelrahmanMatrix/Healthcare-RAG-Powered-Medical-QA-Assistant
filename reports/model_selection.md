# MLflow Model Selection

**Selected run:** `7d1a2bc1`

**Reason:** Highest bleu_rag (0.0157) among 5 runs; latency tie-broken at 1040ms.

**Parameters:**
- `embedding_model`: `pritamdeka/S-PubMedBert-MS-MARCO`
- `inject_k`: `3`
- `llm_model`: `google/flan-t5-base (or Groq meta-llama/llama-4-scout-17b-16e-instruct)`
- `max_context_words`: `200`
- `top_k`: `3`

**Metrics:**
- `avg_latency_ms`: `1040.0000`
- `bleu_baseline`: `0.0008`
- `bleu_improvement_pct`: `1759.2596`
- `bleu_rag`: `0.0157`
- `faiss_index_size`: `211188.0000`
- `macro_f1`: `0.8670`
- `rouge_baseline`: `0.0265`
- `rouge_rag`: `0.1662`