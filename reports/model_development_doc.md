# Model Development Documentation
**Healthcare RAG-Powered Medical Q&A Assistant**
**Owner:** Abdelrahman Mostafa Sayed
**Generated:** 2026-05-23 00:39:24

---

## 1. Architecture Overview
User Query
│
▼
┌─────────────────────────┐
│  BioBERT Classifier   │ → Predicts medical category
└────────────┬────────────┘
│ category
▼
┌─────────────────────────┐
│ FAISS Vector Store │ → Retrieves top-20 candidates
│ (category-prioritised) │ (reranked to top-5 for injection)
│ (category-prioritised) │ (matching category boosted)
└────────────┬────────────┘
│ context chunks
▼
┌─────────────────────────┐
│ meta-llama/llama-4-scout-17b-16e-instruct LLM │ → Generates answer from context
└────────────┬────────────┘
│
▼
┌─────────────────────────┐
│ Medical Disclaimer │ → Appended to every response
└─────────────────────────┘


## 2. Component Details

### 2a. Embedding Model
| Item | Value |
|---|---|
| Model | `pritamdeka/S-PubMedBert-MS-MARCO` |
| Dimension | 768 |
| Purpose | Encode text chunks and queries for semantic similarity |
| Rationale | Biomedical-domain model pre-trained on PubMed/PMC. Superior retrieval precision for medical text vs general-purpose models. |

### 2b. Vector Store
| Item | Value |
|---|---|
| Type | FAISS `IndexFlatIP` |
| Vectors | 209,108 |
| Chunk format | Question + Context + Answer |
| Rationale | Exact search (no approximation errors). IndexFlatIP chosen for correctness at the current corpus scale. |

### 2c. Classifier (Routing Layer)
| Item | Value |
|---|---|
| Model | `dmis-lab/biobert-v1.1` (fine-tuned) |
| Classes | 6 (Symptoms, Diagnosis, Treatment, Medication, Prevention, General) |
| Purpose | Route queries to category-relevant chunks |
| Rationale | BioBERT pre-trained on PubMed/PMC — superior domain fit for medical text classification. Category routing improves retrieval precision. |

### 2d. Language Model
| Item | Value |
|---|---|
| Model | `meta-llama/llama-4-scout-17b-16e-instruct` via Groq API (fallback: `google/flan-t5-base`) |
| Type | Text-to-text generation |
| Max tokens | 256 |
| Rationale | Groq API provides high-quality generation. flan-t5-base used as offline fallback for reproducibility without API keys. |

## 3. Evaluation Methodology

### 3a. Classification
- **Split:** 80/10/10 (train/val/test), stratified
- **Metric:** Macro F1 (target ≥ 78%)
- **Class weights:** Applied via custom WeightedTrainer

### 3b. RAG Pipeline
- **Held-out set:** 2,000 queries NOT in FAISS index
- **Retrieval:** FAISS retrieves top-20 candidates, CrossEncoder reranker scores them, top-inject_k (default 5) injected into LLM prompt
- **Baseline:** Same LLM (meta-llama/llama-4-scout-17b-16e-instruct) without retrieval context
- **Metrics:** BLEU (NLTK), ROUGE-L (rouge-score library)
- **Targets:** ROUGE-L ≥ 0.15, BLEU improvement ≥ +6% (secondary; BERTScore F1 ≥ 0.80 is the primary metric)

### 3c. Hallucination
- **Method:** Manual review of 30 random RAG responses
- **Criteria:** Response contains medical claims not supported by reference or retrieved context
- **Target:** ≤ 15% hallucination rate

## 4. Category Routing Strategy

The classifier doesn't just label queries — it improves retrieval:
1. FAISS retrieves 20 candidates from FAISS (fixed DEFAULT_TOP_K=20)
2. Candidates matching the predicted category are prioritised
3. CrossEncoder reranks the 20 candidates; top-5 are injected into the LLM prompt (DEFAULT_INJECT_K=5)

**Retrieval detail:** The pipeline retrieves 20 candidates from FAISS (`top_k=20`),
**BM25 keyword index:** retrieves the top-k results by BM25 score using a medical-aware tokeniser that preserves hyphenated compound terms. Results are merged with FAISS candidates, deduplicated by chunk ID, and the unified list is re-ranked by the CrossEncoder.
**Note on the FAISS index file:** The file is named `pubmedqa_index_flatl2.faiss`. Although the underlying index type is IndexFlatIP (inner product), the file was saved with the `flatl2` suffix. With L2-normalised embeddings, inner product is equivalent to cosine similarity.
reranks them with a CrossEncoder, and injects the top-`inject_k` (default 5) into the LLM prompt.
All 20 candidates are returned in the API payload for transparency.

**Integrated test results:**
- Queries tested: 10
- Category routing match rate: 84.5%
- All disclaimers present: True

## 5. Design Decisions

| Decision | Rationale |
|---|---|
| Chunk = RecursiveCharacterTextSplitter (700/150) | Maximises semantic signal for retrieval |
| Top-20 candidates (reranked to top-5 for injection) | Balances recall (20 candidates) with context richness (top-5 injected into LLM) |
| Category routing | Improves precision for specialised medical queries |
| Medical disclaimer | Mandatory for responsible AI in healthcare domain |
| Local LLM (no API) | Ensures reproducibility, no cost, no rate limits |

## 6. Known Limitations
- flan-t5-base has limited generation quality compared to larger models
- FAISS IndexFlatIP is exact search — may need IVF for larger datasets
- Category routing depends on classifier accuracy
- Free-tier Azure deployment has cold-start latency
- Medical disclaimer is static — doesn't adapt to confidence level

**Status: M2 Task 5 — Completed ✅**
