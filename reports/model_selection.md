# MLflow Model Selection

> **Scope note:** The BLEU scores in this file (bleu_rag: 0.0157, bleu_baseline: 0.0008, improvement: +1,759%) were recorded during MLflow hyperparameter search experiments using google/flan-t5-base as the comparison baseline — a small local model with near-zero BLEU output. The official submission evaluation in reports/evaluation_report.md uses meta-llama/llama-4-scout-17b-16e-instruct via Groq as the baseline, which is a far stronger model and produces different absolute BLEU values (RAG: 0.0239, baseline: 0.0276, delta: −13.4%). These two documents measure different things. The evaluation_report.md figures are the authoritative submission metrics. This file documents the hyperparameter selection process only.

**Selected run:** `7d1a2bc1`

**Reason:** Lowest avg_latency_ms (1,040ms) among all 5 runs. BLEU scores were nearly identical across runs (0.0154–0.0157), making latency the effective selection criterion.

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