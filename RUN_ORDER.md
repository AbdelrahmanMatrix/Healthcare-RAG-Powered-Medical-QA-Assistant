# 🚀 Run Order — Healthcare RAG Assistant

Run notebooks **in order**. Each one depends on the previous.

## Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables (.env file)
cp .env.example .env
# Edit .env and add:
#   GROQ_API_KEY=your_groq_key_here
#   HF_TOKEN=your_huggingface_token_here
```

---

## Step 1 — Data Pipeline (run once)

| Notebook | What it does | Output |
|---|---|---|
| `01_data_loading.ipynb` | Load PubMedQA from HuggingFace | `data/raw/` |
| `02_preprocessing.ipynb` | Clean, deduplicate, validate | `data/processed/pubmedqa_cleaned.csv` |
| `03_category_labelling.ipynb` | Label 6 medical categories | `data/processed/pubmedqa_labelled.csv` |
| `04_eda.ipynb` | Exploratory analysis + figures | `reports/figures/` |

---

## Step 2 — Embeddings & FAISS (run once)

```
05_embeddings_vectorstore.ipynb
```
- Reserves **1,000 rows** as clean holdout (NOT in FAISS)
- Generates embeddings with **S-PubMedBert-MS-MARCO** (biomedical)
- Builds FAISS index
- Saves `data/embeddings/faiss_index/` and `data/processed/eval_holdout.csv`

---

## Step 3 — RAG Pipeline (run once)

```
06_rag_pipeline.ipynb
```
- Tests retrieval + Groq LLM generation
- Verifies reranker is working

---

## Step 4 — BioBERT Classifier (run once, then upload)

```
07_classification_model.ipynb
```
- Fine-tunes `dmis-lab/biobert-v1.1` on 6 medical categories
- Target: macro F1 ≥ 78% (expected: 90–93%)
- Saves to `models/classifier/biobert_classifier/`

**Then upload weights:**
```bash
huggingface-cli login
python scripts/upload_classifier_to_hub.py
```

---

## Step 5 — Evaluation (key notebook)

```
08_evaluation.ipynb
```
- Loads 200 queries from clean holdout CSV
- Runs RAG vs plain LLM
- Computes: **BLEU**, **ROUGE-L**, **BERTScore** (primary), **Faithfulness**
- Manual hallucination review (30 samples)
- Saves `reports/evaluation_report.md` and `reports/rag_evaluation_results.csv`

---

## Step 6 — Integration & End-to-End

```
09_integrated_pipeline.ipynb
10_end_to_end_test.ipynb
```

---

## Step 7 — Local Docker Test

```bash
make docker-build
docker-compose -f docker/docker-compose.yml up -d
curl http://localhost:8000/health
# Expected: {"status":"ok","model_loaded":true}
```

---

## Step 8 — Azure Deployment (Eman)

Follow `docs/M3_Complete_Guide.md`.

---

## Step 9 — Latency Test (after Azure)

```bash
export AZURE_APP_URL=https://your-azure-url.azurewebsites.net
python scripts/latency_test.py
```

Fill results in `reports/deployment_test_report.md`.

---

## Key Files Changed in This Version (v2)

| File | Change |
|---|---|
| `config/settings.py` | PubMedBERT embedding, reranker, BioBERT classifier path |
| `src/rag/pipeline.py` | Reranker, improved prompt, top_k=10, inject_k=3 |
| `src/rag/embeddings.py` | PubMedBERT default, normalize_embeddings=True |
| `src/evaluation/metrics.py` | Added BERTScore + Faithfulness metrics |
| `src/classification/classifier.py` | BioBERT (AutoTokenizer/AutoModel), fallback to DistilBERT |
| `requirements.txt` | bert-score, rank-bm25 added, torch uncommented |
| `notebooks/05_embeddings_vectorstore.ipynb` | PubMedBERT, holdout=1000, saves eval_holdout.csv |
| `notebooks/07_classification_model.ipynb` | dmis-lab/biobert-v1.1, updated repo ID |
| `notebooks/08_evaluation.ipynb` | Loads holdout CSV, BERTScore, Faithfulness, honest report |
| `scripts/upload_classifier_to_hub.py` | Updated for BioBERT repo |
| `reports/evaluation_report.md` | Honest ROUGE-L note, BERTScore as primary metric |
