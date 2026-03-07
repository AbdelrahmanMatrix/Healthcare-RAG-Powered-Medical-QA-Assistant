# 🏥 Healthcare RAG-Powered Medical Q&A Assistant

> **eyouth × DEPI | Microsoft Machine Learning Track | 2025**

A production-grade Retrieval-Augmented Generation (RAG) system that answers medical questions accurately by grounding every response in verified, peer-reviewed medical knowledge — eliminating hallucination and improving healthcare accessibility in Egypt and the developing world.

---

## 📌 Project Overview

| Field | Details |
|---|---|
| **Track** | Microsoft Machine Learning |
| **Project Type** | Project 5 — Customer Support RAG-Powered Intelligent Chatbot (Healthcare Domain) |
| **Domain** | Healthcare / Medical NLP / RAG |
| **Team Leader** | Abdelrahman Mostafa Sayed |

---

## 🚀 Live Demo

| Resource | Link |
|---|---|
| **GitHub Repository** | [Healthcare-RAG-Powered-Medical-QA-Assistant](https://github.com/AbdooMatrix/Healthcare-RAG-Powered-Medical-QA-Assistant) |
| **Deployed API** | *To be added after Milestone 3 (Azure deployment)* |
| **Streamlit Dashboard** | *To be added after Milestone 4* |

---

## 🗂️ Project Structure

```
Healthcare-RAG-Powered-Medical-QA-Assistant/
│
├── data/
│   ├── raw/                        # Original downloaded datasets
│   ├── processed/                  # Cleaned & preprocessed data
│   └── embeddings/                 # FAISS vector store files
│
├── notebooks/                      # Phase-by-phase Jupyter notebooks
│   ├── 01_data_loading.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_embeddings_vectorstore.ipynb
│   ├── 05_rag_pipeline.ipynb
│   ├── 06_classification_model.ipynb
│   └── 07_evaluation.ipynb
│
├── src/                            # Reusable Python modules
│   ├── data/                       # Loader & preprocessor
│   ├── rag/                        # Embeddings, FAISS, LangChain pipeline
│   ├── classification/             # DistilBERT query classifier
│   └── evaluation/                 # BLEU, ROUGE, F1 metrics
│
├── api/                            # FastAPI REST backend
├── dashboard/                      # Streamlit monitoring dashboard
├── mlops/                          # MLflow experiment tracking
├── docker/                         # Dockerfile & docker-compose
├── tests/                          # Unit tests
├── reports/                        # EDA & evaluation reports
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

| Dataset | Source | Size | Role |
|---|---|---|---|
| **PubMedQA** | HuggingFace (llamafactory) | 11,000 pairs | Primary — peer-reviewed research |
| ChatDoctor-HealthCareMagic-100k | HuggingFace (lavita) | 100,000 pairs | Supplementary — conversational tone |
| medical_meadow_medqa | HuggingFace (medalpaca) | 182,000 pairs | Supplementary — clinical depth |

> **Current Phase:** Working with PubMedQA only (Milestone 1)

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────┐
│  Query          │  DistilBERT classifier
│  Classification │  → Symptoms / Diagnosis / Treatment / Medication / General
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Embedding      │  sentence-transformers (all-MiniLM-L6-v2 / BioBERT)
│  Generation     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FAISS Vector   │  Retrieve top-k most relevant medical Q&A pairs
│  Store Retrieval│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LLM Response   │  LangChain + context injection → grounded response
│  Generation     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Safety Layer   │  Append medical disclaimer to every response
└────────┬────────┘
         │
         ▼
    Final Response
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/AbdooMatrix/Healthcare-RAG-Powered-Medical-QA-Assistant.git
cd Healthcare-RAG-Powered-Medical-QA-Assistant
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
# Fill in your Azure and HuggingFace credentials
```

### 5. Run notebooks in order
```bash
jupyter notebook notebooks/01_data_loading.ipynb
```

---

## 📈 Key Performance Indicators (KPIs)

| Metric | Target |
|---|---|
| Classification F1-Score | ≥ 85% |
| RAG BLEU Score improvement over baseline | ≥ 25% |
| RAG ROUGE-L Score | ≥ 0.45 |
| API Response Latency | ≤ 3000 ms |
| Hallucination Rate | ≤ 5% |
| API Uptime | ≥ 95% |

---

## 👥 Team

| Name | Role | Responsibility |
|---|---|---|
| **Abdelrahman Mostafa Sayed** | Team Leader | RAG pipeline, LangChain integration |
| Ziad Ahmed El-Nady | ML Engineer | Classification model, DistilBERT fine-tuning |
| Youssef George Youssef | Data Engineer | Dataset loading, preprocessing, EDA |
| Doha Khaled Mahmoud | Backend & Deployment | FastAPI, Docker, Azure App Service |
| Eman Khalid Ismail | MLOps & Dashboard | MLflow, Streamlit dashboard, documentation |

---

## 🗓️ Milestones

| Milestone | Phase | Timeline | Status |
|---|---|---|---|
| M1 | Data Collection & Preprocessing | Week 1–2 | 🔄 In Progress |
| M2 | Model Development & Evaluation | Week 3–5 | ⏳ Pending |
| M3 | Azure Deployment | Week 6–7 | ⏳ Pending |
| M4 | MLOps & Dashboard | Week 8–9 | ⏳ Pending |
| M5 | Final Submission | Week 10–12 | ⏳ Pending |

---

## ⚠️ Medical Disclaimer

> This system is an **informational assistant only**. It does not provide diagnosis, prescribe medication, or replace professional medical consultation. Always consult a qualified healthcare professional for medical decisions.

---

## 📄 License

This project is developed as part of the eyouth × DEPI initiative (2025). All datasets used are publicly available and ethically sourced from HuggingFace.