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
| 1    | 01_data_loading.ipynb            | `data/raw/pubmedqa_raw.csv`                      | 9,994 rows loaded |
| 2    | 02_preprocessing.ipynb           | `data/processed/pubmedqa_cleaned.csv`            | Cleaned + normalised |
| 3    | 8-Category_Labeling.ipynb        | `data/processed/pubmedqa_cleaned_Labeled.csv`    | 6 categories added |
| 4    | 03_eda.ipynb                     | `reports/eda_report.md`                          | Full EDA + visuals |

## 3. Category Distribution (Final)
- Symptoms: 3767 (~37.7%)
- Treatment: 2259 (~22.6%)
- General: 2069 (~20.7%)
- Medication: 1337 (~13.4%)
- Diagnosis: 461 (~4.6%)
- Prevention: 101 (~1.0%)

**All 6 categories have ≥ 1% representation** → KPI passed.

## 4. Key Findings from EDA
- Average question length: 13.3 words
- Average context length: 197.2 words  
- Average answer length: 42.3 words
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
jupyter notebook notebooks/8-Category_Labeling.ipynb
jupyter notebook notebooks/03_eda.ipynb