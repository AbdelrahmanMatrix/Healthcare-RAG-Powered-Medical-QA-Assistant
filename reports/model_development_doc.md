# Model Development Documentation
**Healthcare RAG-Powered Medical Q&A Assistant**
**Owner:** Abdelrahman Mostafa Sayed
**Generated:** 2026-04-30 17:47:03

---

## 1. Architecture Overview
User Query
│
▼
┌─────────────────────────┐
│ DistilBERT Classifier │ → Predicts medical category
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
│ flan-t5-base LLM │ → Generates answer from context
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
| Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Dimension | 384 |
| Purpose | Encode text chunks and queries for semantic similarity |
| Rationale | Lightweight, fast inference, strong semantic quality. Good balance between speed and accuracy for a medical Q&A system. |

### 2b. Vector Store
| Item | Value |
|---|---|
| Type | FAISS `IndexFlatL2` |
| Vectors | 9,800 |
| Chunk format | Question + Context + Answer |
| Rationale | Exact search (no approximation errors). IndexFlatL2 chosen for correctness — acceptable for 10K vectors. |

### 2c. Classifier (Routing Layer)
| Item | Value |
|---|---|
| Model | `distilbert-base-uncased` (fine-tuned) |
| Classes | 6 (Symptoms, Diagnosis, Treatment, Medication, Prevention, General) |
| Purpose | Route queries to category-relevant chunks |
| Rationale | Lightweight transformer, fast inference. Category routing improves retrieval precision by prioritising domain-relevant sources. |

### 2d. Language Model
| Item | Value |
|---|---|
| Model | `google/flan-t5-base` |
| Type | Text-to-text generation |
| Max tokens | 256 |
| Rationale | Free, local (no API key needed), instruction-tuned, good at following prompts. Chosen over paid APIs for reproducibility and reliability during demos. |

## 3. Evaluation Methodology

### 3a. Classification
- **Split:** 80/10/10 (train/val/test), stratified
- **Metric:** Macro F1 (target ≥ 78%)
- **Class weights:** Applied via custom WeightedTrainer

### 3b. RAG Pipeline
- **Held-out set:** 200 queries NOT in FAISS index
- **Baseline:** Same LLM (flan-t5-base) without retrieval context
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
- Category routing match rate: 76.0%
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
