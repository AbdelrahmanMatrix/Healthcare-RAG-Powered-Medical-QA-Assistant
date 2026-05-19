# EDA Report — M1 Task 4
**Healthcare RAG-Powered Medical Q&A Assistant**
**Owner:** Doha Khaled Mahmoud
**Generated:** 2026-05-19 19:43:22

---

## Dataset Overview
- **Source:** `pubmedqa_labelled.csv`
- **Total rows:** 211,188
- **Columns:** ['question', 'context', 'answer', 'category', 'question_length', 'context_length', 'answer_length']

## 1. Category Distribution

| Category | Count | Percentage | Flag |
|----------|-------|------------|------|
| Medication | 71,539 | 33.9% |  |
| Treatment | 48,580 | 23.0% |  |
| Diagnosis | 31,919 | 15.1% |  |
| General | 28,252 | 13.4% |  |
| Prevention | 22,179 | 10.5% |  |
| Symptoms | 8,719 | 4.1% |  |

**KPI Check:** All 6 categories present with ≥ 1% representation ✅
**Skew Analysis:** ✅ No critically skewed categories

## 2. Text Length Statistics (Word Count)

| Metric | Question | Context | Answer |
|--------|----------|---------|--------|
| Mean | 15.3 | 197.3 | 37.6 |
| Median | 15.0 | 197.0 | 34.0 |
| Min | 3 | 1 | 1 |
| Max | 109 | 908 | 510 |

## 3. Top 20 Medical Terms

| Term | Frequency |
|------|-----------|
| cells | 172,210 |
| expression | 154,854 |
| levels | 120,421 |
| cell | 112,137 |
| treatment | 94,167 |
| both | 74,582 |
| mice | 73,715 |
| cancer | 73,703 |
| disease | 73,502 |
| protein | 73,359 |
| activity | 63,635 |
| been | 59,733 |
| clinical | 58,320 |
| blood | 57,654 |
| human | 54,240 |
| gene | 54,093 |
| serum | 51,237 |
| role | 50,718 |
| response | 50,569 |
| whether | 48,510 |

## 4. Average Answer Length per Category

| Category | Avg Words |
|----------|-----------|
| General | 39.8 |
| Diagnosis | 38.4 |
| Symptoms | 37.8 |
| Treatment | 37.6 |
| Prevention | 37.0 |
| Medication | 36.5 |

## 5. Key Findings
- Strongest length correlation: context ↔ answer (0.063)
- Dataset is dominated by Symptoms (33.9%) and Diagnosis (23.0%)
- ✅ No critically skewed categories
- Dataset is ready for RAG + Classification training

## 6. Visualisations Produced
1. `01_category_frequency.png` — Category distribution bar chart
2. `02_length_histograms.png` — Question, context, answer length histograms
3. `03_top20_medical_terms_wordcloud.png` — Top 20 terms wordcloud + bar chart
4. `04_avg_answer_length_per_category.png` — Average answer length per category
5. `05_length_boxplot_correlation.png` — Boxplot + correlation heatmap (bonus)

## 7. M1 KPI Status
- [x] Missing values handled ≥ 90%
- [x] Data accuracy after preprocessing ≥ 85%
- [x] All 6 categories present with ≥ 1% each
- [x] EDA report contains all 4 required visualisations
- [ ] Full pipeline reproducible (Task 5)

**Status: M1 Task 4 — Completed ✅**
