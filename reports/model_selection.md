# MLflow Model Selection

**Selected run:** `037cf69f`

**Reason:** Highest bertscore_f1_primary (0.8047) among 6 runs (latency: 2227ms).

BERTScore F1 is the primary quality metric for abstractive RAG systems (Lewis et al. 2020). The value (0.8047) was computed in NB08 and is the authoritative pass/fail metric. BLEU is tracked as a secondary diagnostic metric only.

**Parameters:**
- `embedding_model`: `pritamdeka/S-PubMedBert-MS-MARCO`
- `inject_k`: `3`
- `llm_model`: `meta-llama/llama-4-scout-17b-16e-instruct (via Groq)`
- `max_context_words`: `200`
- `top_k`: `20`

**Metrics:**
- `avg_latency_ms`: `2227.3000`
- `bertscore_f1`: `0.8047`
- `bertscore_f1_primary`: `0.8047`
- `bleu_baseline`: `0.0276`
- `bleu_improvement_pct`: `-13.2400`
- `bleu_rag`: `0.0239`
- `macro_f1`: `0.9100`
- `rouge_baseline`: `0.1788`
- `rouge_improvement_pct`: `5.5300`
- `rouge_rag`: `0.1887`