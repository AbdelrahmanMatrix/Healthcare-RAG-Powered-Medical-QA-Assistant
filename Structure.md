# Healthcare RAG-Powered Medical Q&A Assistant
## Repository Structure — v2 (Final, Aligned with Execution Plan)

> **eyouth × DEPI 2025 | Microsoft Machine Learning Track | Project 5**
> This structure is fully cross-checked against the Milestone Execution Plan.
> Every file listed here is produced by a specific task. No file is orphaned.

---

## Fixes From the Original Structure

| # | Change | Reason |
|---|--------|--------|
| 1 | Added `src/data/labeller.py` | M1 Task 3 (category labelling) produces this file — it was missing |
| 2 | Added `models/classifier/` folder | M2 Task 3 saves the fine-tuned DistilBERT model here — folder was missing |
| 3 | Expanded `reports/` to list all produced documents | Every milestone produces `.md` or `.pdf` docs — none were listed in original |
| 4 | Added `final_presentation.pptx` at root level | M5 Task 2 produces this — no location was defined in the original structure |
| 5 | Removed folder ownership table | Team uses a rotation model — no person owns a folder. See Execution Plan for task assignments |

---

## Full Repository Structure

```
Healthcare-RAG-Powered-Medical-QA-Assistant/
│
├── data/
│   ├── raw/
│   │   └── pubmedqa_raw.csv                  # M1 T1 — raw download, never modified
│   ├── processed/
│   │   ├── pubmedqa_cleaned.csv              # M1 T2 — after cleaning pipeline
│   │   └── pubmedqa_labelled.csv             # M1 T3 — with 'category' column added
│   └── embeddings/
│       └── faiss_index/                      # M2 T1 — 10,000 sentence embeddings
│
├── notebooks/
│   ├── 01_data_loading.ipynb                 # M1 T1 — load PubMedQA, validate schema
│   ├── 02_preprocessing.ipynb                # M1 T2 — clean and normalise
│   ├── 03_eda.ipynb                          # M1 T4 — EDA (4 visualisations)
│   ├── 04_embeddings_vectorstore.ipynb       # M2 T1 — generate embeddings, build FAISS
│   ├── 05_rag_pipeline.ipynb                 # M2 T2 — full LangChain RAG demo
│   ├── 06_classification_model.ipynb         # M2 T3 — DistilBERT fine-tuning
│   └── 07_evaluation.ipynb                   # M2 T4 — BLEU, ROUGE-L, A/B evaluation
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                         # M1 T1 — HuggingFace + CSV loading functions
│   │   ├── preprocessor.py                   # M1 T2 — cleaning functions
│   │   └── labeller.py                       # M1 T3 — keyword-regex category labeller [NEW]
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── embeddings.py                     # M2 T1 — sentence embedding generation
│   │   ├── vectorstore.py                    # M2 T1 — FAISS index setup and retrieval
│   │   └── pipeline.py                       # M2 T2 — full RAG chain + disclaimer layer
│   ├── classification/
│   │   ├── __init__.py
│   │   └── classifier.py                     # M2 T3 — DistilBERT fine-tune + inference
│   └── evaluation/
│       ├── __init__.py
│       └── metrics.py                        # M2 T4 — BLEU, ROUGE-L, F1 helpers
│
├── api/
│   ├── __init__.py
│   ├── main.py                               # M3 T1 — FastAPI app + latency middleware
│   ├── routes/
│   │   ├── __init__.py
│   │   └── query.py                          # M3 T1 — POST /query and GET /health
│   └── schemas/
│       ├── __init__.py
│       └── request.py                        # M3 T1 — Pydantic request/response models
│
├── dashboard/
│   └── app.py                                # M4 T3 — Streamlit KPI dashboard (4 sections)
│
├── mlops/
│   └── mlflow_tracking.py                    # M4 T1 — MLflow experiment tracking setup
│
├── models/
│   └── classifier/                           # M2 T3 — saved DistilBERT weights [NEW]
│
├── docker/
│   ├── Dockerfile                            # M3 T2 — container definition
│   └── docker-compose.yml                    # M3 T2 — local multi-service test
│
├── tests/
│   ├── test_preprocessing.py                 # M1 T5 — tests for preprocessor + labeller
│   ├── test_rag_pipeline.py                  # M2 T5 — tests for FAISS retrieval + pipeline
│   └── test_api.py                           # M3 T1 — tests for /query and /health
│
├── reports/
│   │
│   │   ── Milestone 1 ──
│   ├── schema_validation_report.md           # M1 T1 — raw dataset validation results
│   ├── eda_report.pdf                        # M1 T4 — EDA visualisations export
│   ├── preprocessing_pipeline_doc.md         # M1 T5 — pipeline steps + design decisions
│   │
│   │   ── Milestone 2 ──
│   ├── evaluation_report.pdf                 # M2 T4 — BLEU/ROUGE-L A/B comparison
│   ├── model_development_doc.md              # M2 T5 — architecture decisions + rationale
│   │
│   │   ── Milestone 3 ──
│   ├── deployment_test_report.md             # M3 T4 — 20-query latency test results
│   ├── integration_doc.md                    # M3 T5 — FastAPI → Docker → ACR → Azure
│   │
│   │   ── Milestone 4 ──
│   ├── model_selection.md                    # M4 T2 — best MLflow run rationale
│   ├── monitoring_doc.md                     # M4 T4 — drift thresholds + retraining steps
│   ├── mlops_doc.md                          # M4 T5 — full MLOps setup summary
│   │
│   │   ── Milestone 5 ──
│   └── final_report.pdf                      # M5 T1 — complete project report (15–25 pages)
│
├── .github/
│   └── workflows/
│       └── ci.yml                            # Optional — CI/CD pipeline
│
├── final_presentation.pptx                   # M5 T2 — DEPI template presentation (13 slides)
├── .env                                      # NEVER COMMIT — API keys + Azure config
├── .gitignore                                # Must include .env, models/, __pycache__/
├── requirements.txt                          # M5 T3 — all deps, version-pinned
└── README.md                                 # M5 T3 — setup, run order, API examples, live URL
```

---

## File-to-Milestone-to-Task Index

Every file in this repo is produced by exactly one task. Use this as your checklist.

| File / Folder | Milestone | Task | Owner |
|---|---|---|---|
| `data/raw/pubmedqa_raw.csv` | M1 | T1 | Abdelrahman |
| `notebooks/01_data_loading.ipynb` | M1 | T1 | Abdelrahman |
| `src/data/loader.py` | M1 | T1 | Abdelrahman |
| `reports/schema_validation_report.md` | M1 | T1 | Abdelrahman |
| `data/processed/pubmedqa_cleaned.csv` | M1 | T2 | Ziad |
| `notebooks/02_preprocessing.ipynb` | M1 | T2 | Ziad |
| `src/data/preprocessor.py` | M1 | T2 | Ziad |
| `data/processed/pubmedqa_labelled.csv` | M1 | T3 | Youssef |
| `src/data/labeller.py` | M1 | T3 | Youssef |
| `notebooks/03_eda.ipynb` | M1 | T4 | Doha |
| `reports/eda_report.pdf` | M1 | T4 | Doha |
| `reports/preprocessing_pipeline_doc.md` | M1 | T5 | Eman |
| `tests/test_preprocessing.py` | M1 | T5 | Eman |
| `data/embeddings/faiss_index/` | M2 | T1 | Ziad |
| `notebooks/04_embeddings_vectorstore.ipynb` | M2 | T1 | Ziad |
| `src/rag/embeddings.py` | M2 | T1 | Ziad |
| `src/rag/vectorstore.py` | M2 | T1 | Ziad |
| `notebooks/05_rag_pipeline.ipynb` | M2 | T2 | Youssef |
| `src/rag/pipeline.py` | M2 | T2 | Youssef |
| `notebooks/06_classification_model.ipynb` | M2 | T3 | Doha |
| `src/classification/classifier.py` | M2 | T3 | Doha |
| `models/classifier/` | M2 | T3 | Doha |
| `notebooks/07_evaluation.ipynb` | M2 | T4 | Eman |
| `src/evaluation/metrics.py` | M2 | T4 | Eman |
| `reports/evaluation_report.pdf` | M2 | T4 | Eman |
| `reports/model_development_doc.md` | M2 | T5 | Abdelrahman |
| `tests/test_rag_pipeline.py` | M2 | T5 | Abdelrahman |
| `api/main.py` | M3 | T1 | Youssef |
| `api/routes/query.py` | M3 | T1 | Youssef |
| `api/schemas/request.py` | M3 | T1 | Youssef |
| `tests/test_api.py` | M3 | T1 | Youssef |
| `docker/Dockerfile` | M3 | T2 | Doha |
| `docker/docker-compose.yml` | M3 | T2 | Doha |
| `reports/deployment_test_report.md` | M3 | T4 | Abdelrahman |
| `reports/integration_doc.md` | M3 | T5 | Ziad |
| `mlops/mlflow_tracking.py` | M4 | T1 | Doha |
| `reports/model_selection.md` | M4 | T2 | Eman |
| `dashboard/app.py` | M4 | T3 | Abdelrahman |
| `reports/monitoring_doc.md` | M4 | T4 | Ziad |
| `reports/mlops_doc.md` | M4 | T5 | Youssef |
| `reports/final_report.pdf` | M5 | T1 | Eman |
| `final_presentation.pptx` | M5 | T2 | Abdelrahman |
| `README.md` *(finalised)* | M5 | T3 | Ziad |
| `requirements.txt` *(pinned)* | M5 | T3 | Ziad |
| `reports/` *(audit)* | M5 | T3 | Ziad |

---

## Rotation Summary — Who Owns What Per Milestone

| | M1 | M2 | M3 | M4 | M5 |
|---|---|---|---|---|---|
| **Abdelrahman** | T1: Schema validation | T5: Pipeline integration | T4: API testing | T3: Streamlit dashboard | T2: Presentation deck |
| **Ziad** | T2: Cleaning pipeline | T1: FAISS + embeddings | T5: Integration docs | T4: Retraining strategy | T3: GitHub cleanup |
| **Youssef** | T3: Category labelling | T2: LangChain RAG | T1: FastAPI dev | T5: E2E system test | T4: Demo video |
| **Doha** | T4: EDA report | T3: DistilBERT classifier | T2: Docker | T1: MLflow tracking | T5: Final submission |
| **Eman** | T5: Docs & submission | T4: BLEU/ROUGE eval | T3: Azure deploy | T2: Model registry | T1: Final report |

---

## Ground Rules

**Before writing a single line of code:**

1. **Create this folder structure on GitHub now** — empty folders need a `.gitkeep` file to be tracked by git
2. **Fix the GitHub link** — the DEPI Playbook requires it to be fixed from day 1 and never changed
3. **Add `.gitignore` immediately** — use the Python template and manually add: `.env`, `models/`, `data/raw/`, `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/`
4. **Never commit `.env`** — verify it is in `.gitignore` before the first push

**During development:**

5. **Run notebooks in order** — 01 → 02 → 03 → 04 → 05 → 06 → 07. Each notebook's output is the next one's input. Never run out of order
6. **Each task owner pushes their files before the milestone meeting** — the team reviews together, then the submission is sent to the mentor
7. **Pin `requirements.txt` after each milestone** — run `pip freeze > requirements.txt` and commit it at the end of every milestone, not just M5
8. **`models/classifier/` goes in `.gitignore` for large weight files** — if the model exceeds GitHub's 100MB file limit, use Git LFS or store weights in Azure Blob and reference them

**File naming:**
- All report docs use underscores and lowercase: `eda_report.pdf`, `model_selection.md`
- All notebooks use the numbered prefix: `01_`, `02_` etc — never rename them after creation
- The FAISS index folder is always `data/embeddings/faiss_index/` — the pipeline hardcodes this path

---

*Structure v2 — fully aligned with Milestone Execution Plan | eyouth × DEPI 2025*
