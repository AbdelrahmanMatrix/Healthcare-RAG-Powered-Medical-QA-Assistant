# M3 Deployment Test Report

**Owner:** Abdelrahman Mostafa Sayed
**Milestone:** M3 — Azure Deployment (Task 4: API Integration Testing & Latency Measurement)
**API URL:** `localhost:8000 (local)`
**Test Script:** `notebooks/10_end_to_end_test.ipynb` (integrated pipeline test)
**Status:** ✅ Completed against local FastAPI instance

> **Note:** Azure deployment was not completed for this submission. Latency was measured
> against the local FastAPI instance (warm) using the NB10 integrated test.
> Cold-start calls (7,142ms, 14,871ms, 8,895ms) are excluded per the KPI definition.

---

## Test Configuration

| Item | Value |
|---|---|
| Test queries | 10 (across all 6 medical categories) |
| Measurement | Warm-instance latency only (cold-start excluded per KPI definition) |
| Warm-up | Pipeline components loaded once before test begins |
| KPI threshold | ≤ 5,000ms per warm request |
| Source | `reports/rag_pipeline_test_log.json` (NB10 output) |

---

## Results

| # | Category | Question | Latency (ms) | Disclaimer | Pass |
|---|---|---|---|---|---|
| 01 | Symptoms | What are the early symptoms of type 2 diabetes? | 1,701 | ✓ | ✅ |
| 02 | Diagnosis | How is pneumonia diagnosed in elderly patients? | 1,955 | ✓ | ✅ |
| 03 | Treatment | What are the current treatment options for hypertension? | 3,295 | ✓ | ✅ |
| 04 | Medication | What are the side effects of metformin? | 2,679 | ✓ | ✅ |
| 05 | Prevention | How can cardiovascular disease be prevented through lifestyle changes? | 2,106 | ✓ | ✅ |
| 06 | General | What is the role of the immune system in fighting infections? | 1,758 | ✓ | ✅ |
| 07 | Treatment | Is laparoscopic surgery better than open surgery for prostatectomy? | 1,483 | ✓ | ✅ |
| 08 | Prevention | Can bacterial gastroenteritis lead to irritable bowel syndrome? | 2,572 | ✓ | ✅ |
| 09 | Symptoms | Does naturopathy effectively treat menopausal symptoms? | 1,791 | ✓ | ✅ |
| 10 | Treatment | Is urgent colonoscopy needed for acute diverticular bleeding? | 2,932 | ✓ | ✅ |

---

## KPI Summary

| KPI | Target | Result | Status |
|---|---|---|---|
| Warm latency ≤ 5,000ms | 10/10 | 10/10 | ✅ Pass |
| Disclaimer present in all responses | 10/10 | 10/10 | ✅ Pass |
| Average warm latency | — | 2,227 ms | — |
| Max warm latency | ≤ 5,000ms | 3,295 ms | ✅ Pass |
| Valid JSON returned for all 10 queries | 10/10 | 10/10 | ✅ Pass |

---

## Notes on Cold-Start

Per the project KPI definition (Section 9.2 of the proposal), **cold-start latency is excluded** from all measurements. The pipeline was fully loaded before the test began, ensuring all results reflect warm-instance performance.

Three queries from NB09 (not included above) exhibited cold-start latency:
- Query 5: 7,142ms
- Query 9: 14,871ms
- Query 10: 8,895ms

These are excluded per the KPI definition (cold-start outliers from pipeline initialization).

---

*Healthcare RAG — M3 Task 4 Deployment Test Report | eyouth × DEPI 2026*
