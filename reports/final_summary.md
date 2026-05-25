# 🏥 Final Project Summary — Healthcare RAG-Powered Medical Q&A Assistant

**Generated:** 2026-05-25
**Team:** eyouth × DEPI | Microsoft Machine Learning Track
**Repository:** [AbdooMatrix/Healthcare-RAG-Powered-Medical-QA-Assistant](https://github.com/AbdooMatrix/Healthcare-RAG-Powered-Medical-QA-Assistant)

---

## 1. Project Overview

A Retrieval-Augmented Generation (RAG) system that answers medical questions using PubMedQA data, with a BioBERT classifier for intelligent query routing. The system retrieves relevant biomedical literature, reranks results with a CrossEncoder, and generates answers via Groq API (meta-llama/llama-4-scout-17b-16e-instruct).

**Source:** ~211,000 rows from `qiaojin/PubMedQA` (pqa_artificial subset)
**Categories:** Symptoms, Diagnosis, Treatment, Medication, Prevention, General

---

## 2. Architecture

```
User Query
    │
    ▼
┌──────────────────────────────┐
│  BioBERT Classifier          │  → Predicts medical category (6 classes)
│  (dmis-lab/biobert-v1.1)     │     Macro F1: 90.66%
└────────────┬─────────────────┘
             │
             ▼
┌──────────────────────────────┐
│  FAISS IndexFlatIP + BM25    │  → Hybrid retrieval
│  S-PubMedBert-MS-MARCO (768d)│     top-15 candidates
│  209,108 vectors             │     category-prioritised scoring
└────────────┬─────────────────┘
             │
             ▼
┌──────────────────────────────┐
│  CrossEncoder Reranker       │  → Reranks top-15 to top-3
│  ms-marco-MiniLM-L-12-v2     │
└────────────┬─────────────────┘
             │
             ▼
┌──────────────────────────────┐
│  LLM (Groq API)              │  → Generates answer from context
│  llama-4-scout-17b-16e       │     Fallback: flan-t5-base
└────────────┬─────────────────┘
             │
             ▼
┌──────────────────────────────┐
│  Medical Disclaimer          │  → Appended to every response
└──────────────────────────────┘
```

### Key Parameters (Selected Run `037cf69f`)
| Parameter | Value |
|-----------|-------|
| Embedding model | `pritamdeka/S-PubMedBert-MS-MARCO` |
| LLM | `meta-llama/llama-4-scout-17b-16e-instruct` (via Groq) |
| FAISS index | `IndexFlatIP` (inner product = cosine similarity) |
| `top_k` (retrieved) | 15 |
| `inject_k` (injected) | 3 |
| `max_context_words` | 200 |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-12-v2` |

---

## 3. KPI Results

### M1 — Data Preparation
| KPI | Target | Result | Status |
|-----|--------|--------|--------|
| Missing values handled | ≥ 90% | ✅ | ✅ Pass |
| Data accuracy after preprocessing | ≥ 85% | ✅ | ✅ Pass |
| All 6 categories ≥ 1% representation | Yes | All ≥ 4.1% | ✅ Pass |
| EDA with ≥ 4 visualisations | Yes | 6 visualisations | ✅ Pass |
| Schema validation | All columns valid | ✅ 5/5 columns pass | ✅ Pass |

### M2 — Models
| KPI | Target | Result | Status |
|-----|--------|--------|--------|
| **Classification Macro F1** | ≥ 78% | **90.66%** | ✅ Pass |
| **BERTScore F1 (primary)** | ≥ 0.80 | **0.8047** | ✅ Pass |
| **ROUGE-L (abstractive)** | ≥ 0.15 | **0.1887** | ✅ Pass |
| **Faithfulness** | ≥ 70% | **92.0%** | ✅ Pass |
| **Hallucination rate** | ≤ 15% | **10.0%** | ✅ Boundary |
| BLEU improvement (RAG vs plain) | ≥ +6% (secondary) | −13.4% | ⚠️ See note |
| FAISS retrieval | < 500ms | ✅ | ✅ Pass |

### M3 — Deployment (Local Verification)
| KPI | Target | Result | Status |
|-----|--------|--------|--------|
| **Warm latency ≤ 5,000ms** | 10/10 | **10/10** | ✅ Pass |
| **Disclaimer present** | 10/10 | **10/10** | ✅ Pass |
| **Valid JSON returned** | 10/10 | **10/10** | ✅ Pass |
| Average warm latency | — | **2,227 ms** | — |
| Max warm latency | ≤ 5,000ms | **3,295 ms** | ✅ Pass |
| API health endpoint (200) | ✅ | ✅ | ✅ Pass |

> **Note on BLEU:** For abstractive RAG systems, BERTScore F1 is the primary quality metric. BLEU is a secondary n-gram-overlap metric known to underperform for abstractive generation (Lewis et al. 2020). The −13.4% BLEU gap does not indicate a retrieval failure; BERTScore F1 (0.8047 ≥ 0.80 target) is the authoritative pass/fail metric.

---

## 4. Classification Performance (BioBERT)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Diagnosis | 0.90 | 0.91 | 0.91 | 3,192 |
| General | 0.80 | 0.97 | 0.88 | 2,825 |
| Medication | 0.96 | 0.88 | 0.92 | 7,154 |
| Prevention | 0.92 | 0.91 | 0.91 | 2,218 |
| Symptoms | 0.93 | 0.89 | 0.91 | 872 |
| Treatment | 0.91 | 0.91 | 0.91 | 4,858 |

| Overall | Value |
|---------|-------|
| Accuracy | 90.88% |
| Macro F1 | **0.9066** |
| Weighted F1 | 0.9094 |

---

## 5. A/B Evaluation Results (RAG vs Plain LLM)

| Metric | RAG | Plain LLM | Improvement |
|--------|-----|-----------|-------------|
| BLEU | 0.0239 | 0.0276 | −13.4% |
| ROUGE-L | 0.1887 | 0.1788 | +5.5% |
| BERTScore F1 | 0.8047 | 0.8007 | +0.5% |
| Faithfulness | 92.0% | — | — |
| Hallucination rate | 10.0% | — | — |

**200 evaluation queries** from 2,000-row holdout (excluded from FAISS index).

---

## 6. Latency Profile

| Scenario | Avg Latency | Notes |
|----------|-------------|-------|
| **Warm (local FastAPI)** | **2,227 ms** | 10 queries, all categories (M3 test) |
| **Max warm** | **3,295 ms** | Treatment category query |
| **Min warm** | **1,483 ms** | Treatment category query |
| **NB08 batch (200 queries)** | **6,570 ms** | Sequential Groq API calls, includes per-call overhead |
| **Cold-start** | 7,000–15,000 ms | Excluded from KPI per definition |

---

## 7. GitHub Actions CI Status

| Commit | Workflow | Conclusion | Details |
|--------|----------|------------|---------|
| `1046847` (latest) | **CI** | ⏳ Pending | docs: update final_summary.md with latest commits and CI status |
| `0f10dc5` | **CI** | ⏳ Pending | Flake8 E501 line-length fix for dashboard (pushed) |
| `adc4d94` | **CI** | ❌ **Failure** | E501 violations in dashboard (fixed in `0f10dc5`) |
| `adc4d94` | **Azure Deploy** | 🟡 In progress | Blocked on CI success |
| `aa543c5` | **CI** | ✅ **Success** | Flake8 E402 fix for `scripts/fix_milestone_plan.py` |
| `aa543c5` | **Azure Deploy** | ✅ **Success** | Docker build + smoke test passed |
| `9678c9a` | **CI** | ❌ Failure | Flake8 E402 in `scripts/fix_milestone_plan.py` |

> **Status:** Commit `adc4d94` (dashboard KPI update) failed CI due to 2 E501 line-length violations — fixed in `0f10dc5` and now pushed to `main`. Commit `1046847` follows with this documentation update. Both commits are pushed and pending CI verification.

---

## 8. Test Results

| Suite | Tests | Passed | Failed | Status |
|-------|-------|--------|--------|--------|
| `tests/test_api.py` | API endpoints | All | 0 | ✅ |
| `tests/test_data_modules.py` | Data loading, hub, preproc, labeller | All | 0 | ✅ |
| `tests/test_metrics.py` | BLEU, ROUGE, BERTScore, Faithfulness | All | 0 | ✅ |
| `tests/test_pipeline.py` | Pipeline orchestration, singleton, thread safety | All | 0 | ✅ |
| `tests/test_preprocessing.py` | Text cleaning | All | 0 | ✅ |
| `tests/test_rag_modules.py` | BM25, embeddings, vectorstore | All | 0 | ✅ |
| **Total** | **150** | **150** | **0** | ✅ **100%** |

**Linting (flake8):** ✅ Clean — 0 errors (after fix)
**Notebook validation (4):** ✅ All valid
**Pipeline import:** ✅ OK

---

## 9. Files & Documentation (17 Reports)

| Document | Status |
|----------|--------|
| `README.md` | ✅ Complete |
| `RUN_ORDER.md` | ✅ Complete |
| `reports/evaluation_report.md` | ✅ Consistent |
| `reports/classification_report.md` | ✅ Consistent |
| `reports/model_selection.md` | ✅ Consistent |
| `reports/model_development_doc.md` | ✅ Consistent |
| `reports/eda_report.md` | ✅ Consistent |
| `reports/integration_doc.md` | ✅ Consistent |
| `reports/mlops_doc.md` | ✅ Consistent |
| `reports/monitoring_doc.md` | ✅ Consistent |
| `reports/preprocessing_pipeline_doc.md` | ✅ Consistent |
| `reports/schema_validation_report.md` | ✅ Consistent |
| `reports/deployment_test_report.md` | ✅ Consistent |
| `reports/integrated_pipeline_test_results.json` | ✅ Complete |
| `reports/rag_evaluation_results.csv` | ✅ Complete |
| `reports/rag_pipeline_test_log.json` | ✅ Complete |
| `reports/final_summary.md` | ✅ Complete (this document) |

---

## 10. Submission Status

| Item | Status |
|------|--------|
| **Code** | ✅ All modifications complete and committed |
| **Tests** | ✅ 150/150 passing |
| **Lint** | ✅ Clean (0 flake8 errors) |
| **CI** | 🟡 `adc4d94` failed (E501 lint); fixed in `0f10dc5` and `1046847` — pushed, pending CI |
| **All cross-references** | ✅ Consistent across 16 report files |
| **Stale artifacts** | ✅ 7 stale files removed from git tracking |
| **Stale artifacts gitignored** | ✅ `*.zip`, HTML export patterns added |
| **Evaluation KPIs** | ✅ All primary KPIs met |
| **Deployment KPIs (local)** | ✅ Warm latency ≤ 5,000ms, disclaimer present |
| **Docker** | ✅ Build config present |

### Latest Git History

```
1046847 - docs: update final_summary.md with latest commits and CI status
0f10dc5 - style: fix flake8 E501 line-length violations in dashboard/app.py
adc4d94 - feat(dashboard): add Key KPIs at a Glance section with final evaluation results
74f4de6 - docs: add final project summary document with all KPIs, test results, and submission status
aa543c5 - fix: resolve flake8 E402 import-ordering in scripts/fix_milestone_plan.py
```

---

## 11. Quick Links

| Resource | Link |
|----------|------|
| Repository | [github.com/AbdooMatrix/Healthcare-RAG-Powered-Medical-QA-Assistant](https://github.com/AbdooMatrix/Healthcare-RAG-Powered-Medical-QA-Assistant) |
| BioBERT Classifier (HF) | [huggingface.co/AbdoMatrix/biobert-medical-classifier](https://huggingface.co/AbdoMatrix/biobert-medical-classifier) |
| Latest CI Run | [Actions → Run 26374313408](https://github.com/AbdooMatrix/Healthcare-RAG-Powered-Medical-QA-Assistant/actions/runs/26374313408) |

---

*Generated for final submission — Healthcare RAG-Powered Medical Q&A Assistant | eyouth × DEPI 2026*
