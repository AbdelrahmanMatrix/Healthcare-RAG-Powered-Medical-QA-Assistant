# MLflow Model Selection

**Selected run:** `632096b4`

**Reason:** Highest bleu_rag (0.0239) among 6 runs (latency: 1140ms).

**Parameters:**
- `embedding_model`: `pritamdeka/S-PubMedBert-MS-MARCO`
- `inject_k`: `3`
- `llm_model`: `meta-llama/llama-4-scout-17b-16e-instruct (via Groq)`
- `max_context_words`: `200`
- `top_k`: `20`

**Metrics:**
- `avg_latency_ms`: `1140.0000`
- `bertscore_f1`: `0.0000`
- `bertscore_f1_primary`: `0.0000`
- `bleu_baseline`: `0.0276`
- `bleu_improvement_pct`: `-13.2400`
- `bleu_rag`: `0.0239`
- `faiss_index_size`: `209108.0000`
- `macro_f1`: `0.9100`
- `rouge_baseline`: `0.1788`
- `rouge_improvement_pct`: `5.5300`
- `rouge_rag`: `0.1887`