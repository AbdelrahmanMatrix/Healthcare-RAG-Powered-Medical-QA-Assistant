# 🏥 Healthcare RAG-Powered Medical Q&A Assistant

> A production-grade Retrieval-Augmented Generation (RAG) system that answers medical questions accurately by grounding responses in verified medical knowledge — eliminating hallucination.

---

## 📌 Project Info

| | |
|---|---|
| **Track** | Microsoft Machine Learning — DEPI Graduation Project |
| **Type** | RAG-Powered Intelligent Q&A Chatbot |
| **Domain** | Healthcare / Medical NLP |
| **Status** | 🚧 In Progress |

---

## 🧠 The Idea

Generic LLMs confidently give wrong medical answers. Our system fixes this using RAG:

**Without RAG:** LLM answers from general training → vague, potentially hallucinated

**With RAG:** System retrieves from USMLE-verified knowledge base → accurate, grounded, trustworthy

> *"Our knowledge base is built from US Medical Licensing Exam questions — the same standard used to certify practicing physicians."*

---

## 🗂️ Dataset

| Dataset | Source | Role |
|---|---|---|
| `medalpaca/medical_meadow_medqa` | HuggingFace (USMLE Exams) | Primary knowledge base |

---

## 🛠️ Tech Stack

- **NLP & RAG:** Python, HuggingFace Transformers, LangChain, FAISS
- **Deployment:** FastAPI, Docker, Azure App Service
- **MLOps:** MLflow, Streamlit
- **Version Control:** GitHub

---

## 🗺️ Milestones

- [ ] M1 — Data Collection & Preprocessing
- [ ] M2 — RAG Pipeline & Model Development
- [ ] M3 — Azure Deployment
- [ ] M4 — MLOps & Monitoring Dashboard
- [ ] M5 — Final Report & Presentation

---

## 👥 Team

| Name | Role |
|---|---|
| Abdelrahman Mostafa Sayed *(Team Leader)* | RAG Pipeline & Project Coordination |
| Ziad Ahmed El-Nady | ML Engineer & Model Evaluation |
| Youssef George Youssef | Data Engineering & Preprocessing |
| Doha Khaled Mahmoud | Backend & Azure Deployment |
| Eman Khalid Ismail | MLOps & Dashboard |

---

## ⚠️ Disclaimer

This system is an **informational assistant only**. It does not provide diagnosis, prescribe medication, or replace professional medical consultation. Always consult a qualified healthcare professional for medical decisions.

---

*DEPI Graduation Project — eyouth x Ministry of Communications and Information Technology, Egypt*
