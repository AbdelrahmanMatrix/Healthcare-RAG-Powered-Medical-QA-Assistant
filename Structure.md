Here's your updated `Structure.md`:

```markdown
# Healthcare RAG-Powered Medical Q&A Assistant
## Repository Structure — v3 (Current State: M2 Task 5)

> **eyouth × DEPI 2025 | Microsoft Machine Learning Track | Project 5**
> This structure is fully cross-checked against the Milestone Execution Plan.
> Every file listed here is produced by a specific task. No file is orphaned.

---

## Changes From v2

| # | Change | Reason |
|---|--------|--------|
| 1 | Added `config/` module with `settings.py` | Centralised configuration — no more hardcoded paths scattered across modules |
| 2 | Added `setup.py` at root | Makes `src/` installable via `pip install -e .` — eliminates all import path issues |
| 3 | Added `Makefile` at root | One-command operations: `make test`, `make run`, `make lint` |
| 4 | Added `.env.example` | Onboarding template — teammates copy to `.env` and fill in values |
| 5 | Added `tests/__init__.py` + `tests/conftest.py` | Proper pytest structure with shared fixtures |
| 6 | Renamed `pipeline (3).py` → `pipeline.py` | Spaces + version numbers break imports |
| 7 | Renamed `8-Category_Labeling.ipynb` → `09_category_labeling.ipynb` | Consistent numbered naming with other notebooks |
| 8 | Renamed `pubmedqa_cleaned_Labeled.csv` → `pubmedqa_labelled.csv` | Consistent lowercase naming |
| 9 | Moved `notebooks/data/embeddings/` → `data/embeddings/` | Artifacts don't belong inside notebooks folder |
| 10 | Added `08_integrated_pipeline.ipynb` to notebook list | M2 T5 integration notebook was missing from v2 |
| 11 | Added `reports/figures/` subfolder | EDA visualisations were unlisted in v2 |
| 12 | Added `docs/` with admin/reference documents | Team admin files kept for internal reference |
| 13 | Added `.gitignore` entries for `.idea/` and `*.egg-info/` | IDE and build artifacts must never be committed |

---

## Full Repository Structure

```
Healthcare-RAG-Powered-Medical-QA-Assistant/
│
├── config/                                       # [NEW] Centralised configuration
│   ├── __init__.py
│   └── settings.py                               # All paths, model names, hyperparams
│
├── data/
│   ├── raw/
│   │   └── pubmedqa_raw.csv                      # M1 T1 — raw download, never modified
│   ├── processed/
│   │   ├── pubmedqa_cleaned.csv                  # M1 T2 — after cleaning pipeline
│   │   └── pubmedqa_labelled.csv                 # M1 T3 — with 'category' column added
│   └── embeddings/
│       └── faiss_index/                          # M2 T1 — 10,000 sentence embeddings
│           ├── .gitkeep
│           ├── chunk_mapping.csv
│           ├── chunk_mapping.pkl
│           └── pubmedqa_index_flatl2.faiss
│
├── notebooks/
│   ├── 01_data_loading.ipynb                     # M1 T1 — load PubMedQA, validate schema
│   ├── 02_preprocessing.ipynb                    # M1 T2 — clean and normalise
│   ├── 03_eda.ipynb                              # M1 T4 — EDA (4 visualisations)
│   ├── 04_embeddings_vectorstore.ipynb           # M2 T1 — generate embeddings, build FAISS
│   ├── 05_rag_pipeline.ipynb                     # M2 T2 — full LangChain RAG demo
│   ├── 06_classification_model.ipynb             # M2 T3 — DistilBERT fine-tuning
│   ├── 07_evaluation.ipynb                       # M2 T4 — BLEU, ROUGE-L, A/B evaluation
│   ├── 08_integrated_pipeline.ipynb              # M2 T5 — classifier + RAG end-to-end
│   └── 09_category_labeling.ipynb                # M1 T3 — category labelling exploration
│
├── src/
│   ├── __init__.py
│   ├── pipeline.py                               # M2 T5 — integrated classifier → RAG pipeline
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                             # M1 T1 — HuggingFace + CSV loading functions
│   │   ├── preprocessor.py                       # M1 T2 — cleaning functions
│   │   └── labeller.py                           # M1 T3 — keyword-regex category labeller
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── embeddings.py                         # M2 T1 — sentence embedding generation
│   │   ├── vectorstore.py                        # M2 T1 — FAISS index setup and retrieval
│   │   └── pipeline.py                           # M2 T2 — full RAG chain + disclaimer layer
│   ├── classification/
│   │   ├── __init__.py
│   │   └── classifier.py                         # M2 T3 — DistilBERT fine-tune + inference
│   └── evaluation/
│       ├── __init__.py
│       └── metrics.py                            # M2 T4 — BLEU, ROUGE-L, F1 helpers
│
├── api/                                          # M3 — FastAPI application
│   ├── __init__.py
│   ├── main.py                                   # M3 T1 — FastAPI app + latency middleware
│   ├── routes/
│   │   ├── __init__.py
│   │   └── query.py                              # M3 T1 — POST /query and GET /health
│   └── schemas/
│       ├── __init__.py
│       └── request.py                            # M3 T1 — Pydantic request/response models
│
├── dashboard/                                    # M4 — Streamlit KPI dashboard
│   └── app.py                                    # M4 T3 — 4-section KPI dashboard
│
├── mlops/                                        # M4 — MLOps tooling
│   └── mlflow_tracking.py                        # M4 T1 — MLflow experiment tracking setup
│
├── models/
│   └── classifier/                               # M2 T3 — saved DistilBERT weights
│       ├── .gitkeep
│       └── checkpoints/
│
├── docker/                                       # M3 — Containerisation
│   ├── Dockerfile                                # M3 T2 — container definition
│   └── docker-compose.yml                        # M3 T2 — local multi-service test
│
├── tests/
│   ├── __init__.py                               # [NEW] Proper package init
│   ├── conftest.py                               # [NEW] Shared pytest fixtures
│   ├── test_preprocessing.py                     # M1 T5 — tests for preprocessor + labeller
│   ├── test_rag_pipeline.py                      # M2 T5 — tests for FAISS retrieval + pipeline
│   └── test_api.py                               # M3 T1 — tests for /query and /health
│
├── reports/
│   ├── figures/
│   │   ├── 01_category_label_distribution.png    # M1 T4
│   │   ├── 02_length_histograms.png              # M1 T4
│   │   ├── 03_length_boxplot_correlation.png     # M1 T4
│   │   ├── 04_top_20_medical_terms.png           # M1 T4
│   │   └── 05_avg_output_length.png              # M1 T4
│   │
│   │   ── Milestone 1 ──
│   ├── schema_validation_report.md               # M1 T1 — raw dataset validation results
│   ├── eda_report.md                             # M1 T4 — EDA findings summary
│   ├── eda_length_stats.csv                      # M1 T4 — length statistics export
│   ├── preprocessing_pipeline_doc.md             # M1 T5 — pipeline steps + design decisions
│   │
│   │   ── Milestone 2 ──
│   ├── rag_evaluation_results.csv                # M2 T4 — BLEU/ROUGE-L raw results
│   ├── model_development_doc.md                  # M2 T5 — architecture decisions + rationale
│   │
│   │   ── Milestone 3 (upcoming) ──
│   ├── deployment_test_report.md                 # M3 T4 — 20-query latency test results
│   ├── integration_doc.md                        # M3 T5 — FastAPI → Docker → ACR → Azure
│   │
│   │   ── Milestone 4 (upcoming) ──
│   ├── model_selection.md                        # M4 T2 — best MLflow run rationale
│   ├── monitoring_doc.md                         # M4 T4 — drift thresholds + retraining steps
│   ├── mlops_doc.md                              # M4 T5 — full MLOps setup summary
│   │
│   │   ── Milestone 5 (upcoming) ──
│   └── final_report.pdf                          # M5 T1 — complete project report (15–25 pages)
│
├── docs/                                         # Team admin & reference documents
│   ├── Abdelrahman Mostafa Sayed (abdomostafa20188@gmail.com).docx
│   ├── Azure free credits .pdf
│   ├── DEPI Graduation Project Playbook.pdf
│   ├── Machine Learning Proposal.pdf
│   ├── Microsoft Machine Learning Projects.pdf
│   ├── Milestone_Execution_Plan.docx
│   └── projects_instructions_Applicants.pdf
│
├── .github/                                      # M5 — CI/CD (upcoming)
│   └── workflows/
│       └── ci.yml                                # Optional — linting + test pipeline
│
├── final_presentation.pptx                       # M5 T2 — DEPI template (13 slides, upcoming)
├── .env                                          # NEVER COMMIT — API keys + Azure config
├── .env.example                                  # [NEW] Onboarding template for teammates
├── .gitignore                                    # Includes .env, .idea/, *.egg-info/, models/, etc.
├── Makefile                                      # [NEW] Common commands: test, run, lint, docker
├── requirements.txt                              # All deps — version-pinned at each milestone
├── setup.py                                      # [NEW] Makes src/ installable (pip install -e .)
├── Structure.md                                  # This file
└── README.md                                     # Project overview, setup, run order, API examples
```

---

## What Exists Now vs What's Upcoming

| Status | Folders / Files |
|--------|----------------|
| ✅ **Complete (M1)** | `data/raw/`, `data/processed/`, `notebooks/01–03`, `notebooks/09`, `src/data/`, `reports/` (M1 docs + figures) |
| ✅ **Complete (M2 T1–T4)** | `data/embeddings/`, `notebooks/04–07`, `src/rag/`, `src/classification/`, `src/evaluation/`, `models/classifier/` |
| 🔄 **In Progress (M2 T5)** | `src/pipeline.py`, `notebooks/08`, `reports/model_development_doc.md`, `tests/test_rag_pipeline.py` |
| ⏳ **M3 (upcoming)** | `api/`, `docker/`, `reports/deployment_test_report.md`, `reports/integration_doc.md` |
| ⏳ **M4 (upcoming)** | `dashboard/`, `mlops/`, `reports/model_selection.md`, `reports/monitoring_doc.md`, `reports/mlops_doc.md` |
| ⏳ **M5 (upcoming)** | `final_presentation.pptx`, `reports/final_report.pdf`, `.github/workflows/`, README finalised |
| 🔧 **Infrastructure (added now)** | `config/`, `setup.py`, `Makefile`, `.env.example`, `tests/conftest.py`, `tests/__init__.py` |

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
| `notebooks/09_category_labeling.ipynb` | M1 | T3 | Youssef |
| `notebooks/03_eda.ipynb` | M1 | T4 | Doha |
| `reports/eda_report.md` | M1 | T4 | Doha |
| `reports/eda_length_stats.csv` | M1 | T4 | Doha |
| `reports/figures/*` | M1 | T4 | Doha |
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
| `reports/rag_evaluation_results.csv` | M2 | T4 | Eman |
| `src/pipeline.py` | M2 | T5 | Abdelrahman |
| `notebooks/08_integrated_pipeline.ipynb` | M2 | T5 | Abdelrahman |
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

---

## Infrastructure Files (No Milestone — Project-Wide)

| File | Purpose | Added |
|---|---|---|
| `config/settings.py` | Centralised paths, model names, hyperparams via pydantic-settings | M2 T5 |
| `config/__init__.py` | Package init | M2 T5 |
| `setup.py` | Makes `src/` installable — `pip install -e .` | M2 T5 |
| `Makefile` | Common commands: `make test`, `make run`, `make lint`, `make docker-build` | M2 T5 |
| `.env.example` | Template for `.env` — teammates copy and fill in values | M2 T5 |
| `tests/__init__.py` | Makes tests/ a proper Python package | M2 T5 |
| `tests/conftest.py` | Shared pytest fixtures (sample data, mock models) | M2 T5 |
| `.gitignore` | Excludes `.env`, `.idea/`, `*.egg-info/`, `__pycache__/`, `.ipynb_checkpoints/` | Day 1 |
| `Structure.md` | This file — repository map | Day 1 |

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

1. **This folder structure is now live on GitHub** — empty folders use `.gitkeep` to be tracked by git
2. **The GitHub link is fixed** — the DEPI Playbook requires it to be fixed from day 1 and never changed
3. **`.gitignore` is active** — includes `.env`, `.idea/`, `*.egg-info/`, `models/`, `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/`
4. **Never commit `.env`** — use `.env.example` as the onboarding template

**During development:**

5. **Run notebooks in order** — 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08. Each notebook's output is the next one's input. Never run out of order
6. **Use `config/settings.py` for all paths and constants** — no hardcoded strings in source files
7. **Install the project** — run `pip install -e .` so all `from src.rag.pipeline import ...` imports work everywhere
8. **Each task owner pushes their files before the milestone meeting** — the team reviews together, then the submission is sent to the mentor
9. **Pin `requirements.txt` after each milestone** — run `pip freeze > requirements.txt` and commit it at the end of every milestone, not just M5
10. **`models/classifier/` goes in `.gitignore` for large weight files** — if the model exceeds GitHub's 100MB file limit, use Git LFS or store weights in Azure Blob and reference them

**File naming:**
- All report docs use underscores and lowercase: `eda_report.md`, `model_selection.md`
- All notebooks use the numbered prefix: `01_`, `02_` etc — never rename them after creation
- The FAISS index folder is always `data/embeddings/faiss_index/` — referenced via `config/settings.py`

---

*Structure v3 — aligned with Milestone Execution Plan + professional infrastructure | eyouth × DEPI 2025*
```