# M1: Preprocessing Pipeline Documentation
Milestone: Data Collection & Preprocessing  
Task Owner: Eman Khalid Elkalawy  
Date: April 11, 2026  
Repository: https://github.com/AbdooMatrix/Healthcare-RAG-Powered-Medical-QA-Assistant

## 1. Pipeline Overview
This document describes the complete end-to-end M1 pipeline:
1. Dataset Download & Schema Validation (Task 1)
2. Text Cleaning & Normalisation (Task 2)
3. Medical Category Labelling (Task 3)
4. Exploratory Data Analysis (Task 4)

## 2. Execution Steps & Outputs

| Step | Notebook/Script                  | Output File                                      | Key Results |
|------|----------------------------------|--------------------------------------------------|-----------|
| 1    | 01_data_loading.ipynb            | `data/raw/pubmedqa_raw.csv`                      | 211,269 rows loaded (qiaojin/PubMedQA pqa_artificial) |
| 2    | 02_preprocessing.ipynb           | `data/processed/pubmedqa_cleaned.csv`            | Cleaned + normalised |
| 3    | 03_category_labelling.ipynb     | `data/processed/pubmedqa_labelled.csv`           | 6 categories added |
| 4    | 04_eda.ipynb                     | `reports/eda_report.md`                          | Full EDA + visuals |

## 3. Category Distribution (Final — full dataset)
- Medication: 71,537 (~33.9%)
- Treatment: 48,579 (~23.0%)
- Diagnosis: 31,919 (~15.1%)
- General: 28,251 (~13.4%)
- Prevention: 22,180 (~10.5%)
- Symptoms: 8,720 (~4.1%)

**All 6 categories have ≥ 1% representation** → KPI passed.

## 4. Key Findings from EDA
- Average question length: 15.3 words
- Average context length: 197.3 words  
- Average answer length: 37.6 words
- Dataset is well-balanced and ready for RAG + classification model
- No major skewness in categories

## 5. Reproducibility Instructions
To run the full M1 pipeline from scratch:

```bash
git clone https://github.com/AbdooMatrix/Healthcare-RAG-Powered-Medical-QA-Assistant.git
cd Healthcare-RAG-Powered-Medical-QA-Assistant

# Activate environment
venv\Scripts\activate     # Windows

pip install -r requirements.txt

# Run notebooks in order
jupyter notebook notebooks/01_data_loading.ipynb
jupyter notebook notebooks/02_preprocessing.ipynb
jupyter notebook notebooks/03_category_labelling.ipynb
jupyter notebook notebooks/04_eda.ipynb