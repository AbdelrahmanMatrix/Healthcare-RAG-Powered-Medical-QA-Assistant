# M3 Integration Documentation

**Owner:** Ziad Ahmed El-Nady | eyouth × DEPI 2025
**Milestone:** M3 — Azure Deployment
**Date:** May 2026

---

## Live URL

> **Deployment status:** Azure deployment was not completed for this submission. All M3 KPIs (warm latency ≤ 5,000ms and disclaimer presence) were verified against a local FastAPI instance, as documented in reports/deployment_test_report.md.
>
> Once deployed, the production endpoint will be:
> `https://healthcare-rag-app.azurewebsites.net`

---

## API Endpoints

| Endpoint | Method | Request Body | Returns |
|---|---|---|---|
| `/query` | POST | `{"question": "...", "top_k": null, "category": null}` | `{answer, category, retrieved_sources, source_citations, disclaimer}` |
| `/health` | GET | — | `{status, model_loaded, classifier_ready, groq_configured, index_vectors}` |
| `/docs` | GET | — | Swagger UI (interactive API docs) |
| `/` | GET | — | `{project, docs, health, version}` |

### Sample `/query` Request

```bash
# Example shown for production format. For local testing, replace with http://localhost:8000
curl -X POST https://healthcare-rag-app.azurewebsites.net/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the symptoms of type 2 diabetes?"}'
```

### Sample `/query` Response

```json
{
  "answer": "Type 2 diabetes symptoms include increased thirst, frequent urination, fatigue, blurred vision, and slow wound healing.",
  "category": "Symptoms",
  "retrieved_sources": ["42", "137", "891", "204", "556"],
  "source_citations": [
    {
      "chunk_id": "42",
      "question": "What symptoms are associated with type 2 diabetes mellitus?",
      "category": "Symptoms",
      "distance": 0.1234,
      "relevance_score": 0.87,
      "excerpt": "Diabetes mellitus type 2 is characterised by elevated blood glucose..."
    }
  ],
  "disclaimer": "This is an informational assistant only. Always consult a qualified healthcare professional for medical decisions. Do not use this information as a substitute for professional medical advice, diagnosis, or treatment."
}
```

---

## Deployment Architecture

```
Client (HTTP/JSON)
    │
    ▼
FastAPI (api/main.py)
    │  ├─ CORS middleware
    │  ├─ Latency middleware (X-Response-Time-Ms header)
    │  └─ Auth middleware (X-API-Key, disabled in dev)
    │
    ▼
api/routes/query.py  →  run_pipeline (src/pipeline.py)
    │
    ├─▶ BioBERT Classifier (src/classification/classifier.py)
    │       └─ Downloads weights from HuggingFace Hub on first request
    │          (huggingface.co/AbdoMatrix/biobert-medical-classifier)
    │
    ├─▶ FAISS Vector Store (src/rag/vectorstore.py)
    │       └─ data/embeddings/faiss_index/pubmedqa_index_flatl2.faiss
    │          (209,108 vectors, S-PubMedBert-MS-MARCO embeddings)
    │
    ├─▶ BM25 Retriever (src/rag/bm25_retriever.py)  [hybrid retrieval]
    │
    └─▶ LLM (Groq meta-llama/llama-4-scout-17b-16e-instruct via openai client)
            └─ Generates answer from top-3 reranked chunks (retrieved from top-20 FAISS candidates)

    │
    ▼
QueryResponse  →  disclaimer injected from config.settings.disclaimer
    │
    ▼
Docker Container (docker/Dockerfile)
    │  └─ python:3.10-slim, port 8000, uvicorn 1 worker
    │
    ▼
Azure Container Registry (healthcareragacr.azurecr.io)
    │
    ▼
Azure App Service (Free Tier F1, Linux)
    └─ healthcare-rag-app.azurewebsites.net
```

---

## Environment Variables in Azure

| Variable | Value | Purpose |
|---|---|---|
| `WEBSITE_PORT` | `8000` | Port FastAPI listens on |
| `PYTHONUNBUFFERED` | `1` | Real-time stdout logs in Azure |
| `HF_TOKEN` | `hf_xxx...` | Downloads BioBERT classifier weights from HuggingFace |
| `GROQ_API_KEY` | `gsk_xxx...` | LLM generation via Groq API (meta-llama/llama-4-scout-17b-16e-instruct) |
| `DEPLOY_ENV` | `Azure App Service F1` | Shown in Streamlit dashboard |
| `DEPLOY_DATE` | `May 2026` | Shown in Streamlit dashboard |

> `.env` is gitignored — never commit it. Copy `.env.example` → `.env` and fill in your values.

---

## Redeployment Steps

When source code changes, redeploy with:

```bash
# Build new image version
docker build -f docker/Dockerfile -t healthcareragacr.azurecr.io/healthcare-rag:v2 .

# Push to ACR
docker push healthcareragacr.azurecr.io/healthcare-rag:v2

# Update Azure Web App to use new image
az webapp config container set \
  --name healthcare-rag-app \
  --resource-group healthcare-rag-rg \
  --docker-custom-image-name healthcareragacr.azurecr.io/healthcare-rag:v2

# Restart app
az webapp restart \
  --name healthcare-rag-app \
  --resource-group healthcare-rag-rg
```

---

## Azure Free Tier Limitations

| Constraint | Value | Impact |
|---|---|---|
| RAM | 1 GB | Model loading is memory-constrained — 1 uvicorn worker only |
| CPU minutes/day | 60 | High-volume testing will exhaust daily quota |
| Idle timeout | 20 min | Cold-start latency 60–120s after idle period |
| Cold-start | 60–120s | Excluded from KPI latency measurement (warm-instance only) |
| Custom domain / HTTPS | HTTPS included | azurewebsites.net URL includes TLS |

**Note:** Keep the Azure URL live until July 2026 — it is required for M4 end-to-end testing, M5 demo video, and the final DEPI submission panel.

---

## KPI Verification

| KPI | Target | How to Verify |
|---|---|---|
| Local API returns 200 | ✅ | `curl http://localhost:8000/health` (verified locally — Azure URL pending deployment) |
| All 20 queries return disclaimer | 20/20 | `python scripts/latency_test.py` → Disclaimer OK |
| Warm latency ≤ 5,000ms | 20/20 | `python scripts/latency_test.py` → Latency pass |
| Stable Docker deployment | ≥ 1 release | Azure portal → App Service → Deployment logs |
| API uptime ≥ 90% over 48h | Pending | Azure deployment not completed — verified locally only |

---

*Healthcare RAG — M3 Integration Documentation | eyouth × DEPI 2025*
