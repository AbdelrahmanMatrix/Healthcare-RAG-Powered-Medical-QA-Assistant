# Model Development Documentation
**Healthcare RAG-Powered Medical Q&A Assistant**
**Owner:** Abdelrahman Mostafa Sayed
**Generated:** 2026-05-21 14:59:01

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
│ FAISS Vector Store │ → Retrieves top-5 chunks
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
| Type | FAISS `IndexFlatL2` |
| Vectors | 210,186 |
| Chunk format | Question + Context + Answer |
| Rationale | Exact search (no approximation errors). IndexFlatL2 chosen for correctness at the current corpus scale. |

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
| Model | `llama-3.1-8b-instant` via Groq API (fallback: `google/flan-t5-base`) |
| Type | Text-to-text generation |
| Max tokens | 256 |
| Rationale | Groq API provides high-quality generation. flan-t5-base used as offline fallback for reproducibility without API keys. |

## 3. Evaluation Methodology

### 3a. Classification
- **Split:** 80/10/10 (train/val/test), stratified
- **Metric:** Macro F1 (target ≥ 78%)
- **Class weights:** Applied via custom WeightedTrainer

### 3b. RAG Pipeline
- **Held-out set:** 200 queries NOT in FAISS index
- **Baseline:** Same LLM (meta-llama/llama-4-scout-17b-16e-instruct) without retrieval context
- **Metrics:** BLEU (NLTK), ROUGE-L (rouge-score library)
- **Targets:** ROUGE-L ≥ 0.38, BLEU improvement ≥ 20%

### 3c. Hallucination
- **Method:** Manual review of 30 random RAG responses
- **Criteria:** Response contains medical claims not supported by reference or retrieved context
- **Target:** ≤ 15% hallucination rate

## 4. Category Routing Strategy

The classifier doesn't just label queries — it improves retrieval:
1. FAISS retrieves 3× more candidates than needed
2. Candidates matching the predicted category are prioritised
3. Top-5 results returned (category matches first, then by distance)

**Integrated test results:**
- Queries tested: 10
- Category routing match rate: 84.5%
- All disclaimers present: True

## 5. Design Decisions

| Decision | Rationale |
|---|---|
| Chunk = Q + Context + Answer | Maximises semantic signal for retrieval |
| Top-5 retrieval | Balances context richness with prompt length limits |
| Category routing | Improves precision for specialised medical queries |
| Medical disclaimer | Mandatory for responsible AI in healthcare domain |
| Local LLM (no API) | Ensures reproducibility, no cost, no rate limits |

## 6. Known Limitations
- flan-t5-base has limited generation quality compared to larger models
- FAISS IndexFlatL2 is exact search — may need IVF for larger datasets
- Category routing depends on classifier accuracy
- Free-tier Azure deployment has cold-start latency
- Medical disclaimer is static — doesn't adapt to confidence level

**Status: M2 Task 5 — Completed ✅**
