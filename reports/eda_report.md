# EDA Report — M1 Task 4
**Healthcare RAG-Powered Medical Q&A Assistant**
**Owner:** Doha Khaled Mahmoud
**Generated:** 2026-05-20 00:05:48

---

## Dataset Overview
- **Source:** `pubmedqa_labelled.csv`
- **Total rows:** 211,186
- **Columns:** ['question', 'context', 'answer', 'category', 'question_length', 'context_length', 'answer_length']

## 1. Category Distribution

| Category | Count | Percentage | Flag |
|----------|-------|------------|------|
| Medication | 71,537 | 33.9% |  |
| Treatment | 48,579 | 23.0% |  |
| Diagnosis | 31,919 | 15.1% |  |
| General | 28,251 | 13.4% |  |
| Prevention | 22,180 | 10.5% |  |
| Symptoms | 8,720 | 4.1% |  |

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
| cells | 172,201 |
| expression | 154,842 |
| levels | 120,417 |
| cell | 112,130 |
| treatment | 94,172 |
| both | 74,581 |
| mice | 73,722 |
| cancer | 73,706 |
| disease | 73,496 |
| protein | 73,352 |
| activity | 63,639 |
| been | 59,730 |
| clinical | 58,319 |
| blood | 57,650 |
| human | 54,233 |
| gene | 54,083 |
| serum | 51,236 |
| role | 50,717 |
| response | 50,577 |
| whether | 48,512 |

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
