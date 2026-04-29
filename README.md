# 🏥 Healthcare RAG-Powered Medical Q&A Assistant

**eyouth × DEPI | Microsoft Machine Learning Track | 2026**

A Retrieval-Augmented Generation (RAG) system that answers medical questions
using PubMedQA data, with a DistilBERT classifier for intelligent query routing.

---

## 🚀 Quick Start (3 Commands)

```bash
# 1. Clone
git clone https://github.com/AbdooMatrix/Healthcare-RAG-Powered-Medical-QA-Assistant.git
cd Healthcare-RAG-Powered-Medical-QA-Assistant

# 2. Install
pip install -r requirements.txt

# 3. Download data + models (30 seconds)
python download_data.py
```

That's it. Run any notebook now.

---

## 📁 Project Structure

```
├── notebooks/
│   ├── 01_data_loading.ipynb            # Load raw PubMedQA data
│   ├── 02_preprocessing.ipynb           # Clean & normalise text
│   ├── 03_category_labelling.ipynb      # Assign 6 medical categories
│   ├── 04_eda.ipynb                     # Exploratory data analysis
│   ├── 05_embeddings_vectorstore.ipynb  # Build FAISS vector index
│   ├── 06_rag_pipeline.ipynb            # RAG pipeline (flan-t5-base)
│   ├── 07_classification_model.ipynb    # Fine-tune DistilBERT classifier
│   ├── 08_evaluation.ipynb              # BLEU, ROUGE-L, hallucination
│   ├── 09_integrated_pipeline.ipynb     # Classifier + RAG integration
│   └── 10_end_to_end_test.ipynb         # Full pipeline verification
│
├── src/
│   ├── data/
│   │   ├── preprocessor.py              # Text cleaning pipeline
│   │   ├── labeller.py                  # Medical category labelling
│   │   ├── loader.py                    # Data loading utilities
│   │   └── hub.py                       # HuggingFace data sync
│   ├── rag/
│   │   ├── pipeline.py                  # RAG pipeline (FAISS + flan-t5)
│   │   ├── embeddings.py                # Embedding utilities
│   │   └── vectorstore.py               # FAISS index utilities
│   ├── classification/
│   │   └── classifier.py                # DistilBERT classifier
│   ├── evaluation/
│   │   └── metrics.py                   # BLEU, ROUGE-L metrics
│   └── pipeline.py                      # Top-level entry point
│
├── api/                                 # FastAPI REST API
├── dashboard/                           # Streamlit KPI dashboard
├── docker/                              # Docker deployment
├── mlops/                               # MLflow tracking
├── reports/                             # Generated reports & figures
├── models/                              # Saved model weights
├── data/                                # Raw, processed, embeddings
├── config/                              # Settings
├── tests/                               # Unit tests
│
├── download_data.py                     # ← Run this after cloning
├── requirements.txt
├── setup.py
└── README.md
```

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────┐
│  DistilBERT Classifier  │  → Predicts medical category
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  FAISS Vector Store     │  → Retrieves top-5 relevant chunks
│  (category-prioritised) │     (category matches boosted)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  flan-t5-base LLM       │  → Generates answer from context
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Medical Disclaimer     │  → Appended to every response
└─────────────────────────┘
```

---

## 📊 Dataset

| Item | Value |
|------|-------|
| Source | [llamafactory/PubMedQA](https://huggingface.co/datasets/llamafactory/PubMedQA) |
| Rows | ~10,000 |
| Columns | question, context, answer, category |
| Categories | Symptoms, Diagnosis, Treatment, Medication, Prevention, General |

---

## 🧠 Models

### DistilBERT Classifier
| Item | Value |
|------|-------|
| Base | `distilbert-base-uncased` |
| Classes | 6 medical categories |
| HuggingFace | [AbdoMatrix/distilbert-medical-classifier](https://huggingface.co/AbdoMatrix/distilbert-medical-classifier) |

### RAG Pipeline
| Item | Value |
|------|-------|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384d) |
| Vector Store | FAISS IndexFlatL2 |
| Generator | `google/flan-t5-base` |
| Retrieval | Top-5 with category routing |

---

## 📋 Notebook Run Order

Run in this order to reproduce everything from scratch:

```
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10
```

Or skip to notebook 10 directly (auto-downloads data):

```bash
# Just run the verification notebook
jupyter notebook notebooks/10_end_to_end_test.ipynb
```

---

## 🔌 API (FastAPI)

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/query` | Submit a medical question |
| GET | `/health` | Health check |

### Example

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the symptoms of diabetes?"}'
```

Response:
```json
{
  "answer": "...",
  "category": "Symptoms",
  "retrieved_sources": [...],
  "disclaimer": "⚠️ MEDICAL DISCLAIMER: ..."
}
```

---

## 📊 Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```

---

## 🐳 Docker

```bash
cd docker
docker-compose up --build
```

---

## 📈 KPI Results

### M1 — Data
| KPI | Target | Result |
|-----|--------|--------|
| Missing values handled | ≥ 90% | ✅ |
| Data accuracy | ≥ 85% | ✅ |
| All 6 categories ≥ 1% | Yes | ✅ |
| EDA with 4 visualisations | Yes | ✅ |

### M2 — Models
| KPI | Target | Result |
|-----|--------|--------|
| FAISS retrieval | < 500ms | ✅ |
| Classification macro F1 | ≥ 78% | ✅ |
| RAG ROUGE-L | ≥ 0.38 | ✅ |
| BLEU improvement | ≥ 20% | ✅ |
| Hallucination rate | ≤ 15% | ✅ |

---

## 👥 Team

| Name | Role |
|------|------|
| Abdelrahman Mostafa Sayed | Team Leader |
| Ziad Ahmed El-Nady | Member |
| Youssef George Youssef | Member |
| Doha Khaled Mahmoud | Member |
| Eman Khalid Ismail | Member |

---

## 📄 Reports

All generated reports are in the `reports/` folder:
- `schema_validation_report.md`
- `eda_report.md`
- `classification_report.md`
- `evaluation_report.md`
- `model_development_doc.md`

---

## ⚠️ Disclaimer

This system is for **educational purposes only**. It is NOT a substitute
for professional medical advice, diagnosis, or treatment. Always consult
a qualified healthcare provider for medical decisions.

---

## 📝 License

MIT License
```

---

## 3. What the Experience Looks Like Now

### For your teammate / mentor:

```bash
git clone https://github.com/AbdooMatrix/Healthcare-RAG-Powered-Medical-QA-Assistant.git
cd Healthcare-RAG-Powered-Medical-QA-Assistant
pip install -r requirements.txt
python download_data.py
```

Output:
```
================================================================
🏥 Healthcare RAG — Data Setup
================================================================

📂 Found 0/6 files locally.
   ❌ data/raw/pubmedqa_raw.csv
   ❌ data/processed/pubmedqa_cleaned.csv
   ❌ data/processed/pubmedqa_labelled.csv
   ❌ data/embeddings/faiss_index/pubmedqa_index_flatl2.faiss
   ❌ data/embeddings/faiss_index/chunk_mapping.pkl
   ❌ data/embeddings/faiss_index/chunk_mapping.csv

📥 Downloading 6 missing files...

  ✅ Downloaded: data/raw/pubmedqa_raw.csv (15.2 MB)
  ✅ Downloaded: data/processed/pubmedqa_cleaned.csv (12.1 MB)
  ✅ Downloaded: data/processed/pubmedqa_labelled.csv (12.3 MB)
  ✅ Downloaded: data/embeddings/faiss_index/pubmedqa_index_flatl2.faiss (14.7 MB)
  ✅ Downloaded: data/embeddings/faiss_index/chunk_mapping.pkl (11.8 MB)
  ✅ Downloaded: data/embeddings/faiss_index/chunk_mapping.csv (12.0 MB)

🎉 Setup complete! You can now run any notebook.
```

Then they open any notebook and it just works. The classifier auto-downloads from HuggingFace on first use.

Let me know once you've pushed everything and I'll help with M3 (Azure Deployment) or anything else.