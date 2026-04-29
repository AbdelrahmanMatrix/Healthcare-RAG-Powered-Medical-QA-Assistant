# EDA Report — M1 Task 4
**Healthcare RAG-Powered Medical Q&A Assistant**
**Owner:** Doha Khaled Mahmoud
**Generated:** 2026-04-29 21:16:54

---

## Dataset Overview
- **Source:** `pubmedqa_labelled.csv`
- **Total rows:** 10,000
- **Columns:** ['question', 'context', 'answer', 'category', 'question_length', 'context_length', 'answer_length']

## 1. Category Distribution

| Category | Count | Percentage | Flag |
|----------|-------|------------|------|
| Symptoms | 5,893 | 58.9% |  |
| Diagnosis | 3,043 | 30.4% |  |
| Treatment | 330 | 3.3% |  |
| Medication | 295 | 2.9% |  |
| General | 290 | 2.9% |  |
| Prevention | 149 | 1.5% |  ⚠️ borderline |

**KPI Check:** All 6 categories present with ≥ 1% representation ✅
**Skew Analysis:** ⚠️ Borderline categories (< 2%): ['Prevention']

## 2. Text Length Statistics (Word Count)

| Metric | Question | Context | Answer |
|--------|----------|---------|--------|
| Mean | 13.3 | 197.3 | 42.3 |
| Median | 13.0 | 195.0 | 39.0 |
| Min | 3 | 26 | 6 |
| Max | 45 | 606 | 313 |

## 3. Top 20 Medical Terms

| Term | Frequency |
|------|-----------|
| treatment | 4,354 |
| clinical | 4,097 |
| health | 3,511 |
| disease | 3,440 |
| care | 3,265 |
| both | 3,176 |
| patient | 3,145 |
| whether | 3,078 |
| cancer | 3,068 |
| surgery | 3,060 |
| those | 2,959 |
| only | 2,762 |
| levels | 2,733 |
| been | 2,567 |
| other | 2,511 |
| factors | 2,499 |
| when | 2,468 |
| followup | 2,242 |
| blood | 2,168 |
| studies | 2,143 |

## 4. Average Answer Length per Category

| Category | Avg Words |
|----------|-----------|
| General | 45.8 |
| Treatment | 43.1 |
| Diagnosis | 42.6 |
| Medication | 42.2 |
| Symptoms | 42.0 |
| Prevention | 41.2 |

## 5. Key Findings
- Strongest length correlation: context ↔ answer (0.147)
- Dataset is dominated by Symptoms (58.9%) and Diagnosis (30.4%)
- ⚠️ Borderline categories (< 2%): ['Prevention']
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
