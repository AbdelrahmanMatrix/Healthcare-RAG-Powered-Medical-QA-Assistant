# Azure Deployment Guide

## Prerequisites

1. **Azure CLI** installed → [Install guide](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli)
2. **Docker** installed and running
3. **`.env` file** with secrets filled in (see `.env.example`)

```bash
cp .env.example .env
# Fill in GROQ_API_KEY and HF_TOKEN
```

---

## One-Command Deploy

```bash
# 1. Login to Azure
az login

# 2. Deploy everything
bash azure/deploy.sh
```

The script:
- Creates Resource Group, Container Registry, App Service Plan, and Web App
- Builds and pushes the Docker image
- Sets all environment variables securely (no secrets in code)
- Prints the live URL when done

**First startup takes 2–5 minutes** — the container downloads the FAISS index from HuggingFace.

---

## Test Your Deployment

```bash
bash azure/test_deployment.sh https://healthcare-rag-app.azurewebsites.net
```

Or manually:

```bash
# Health check
curl https://healthcare-rag-app.azurewebsites.net/health

# Ask a medical question
curl -X POST https://healthcare-rag-app.azurewebsites.net/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Does aspirin reduce cardiovascular risk?"}'

# Swagger UI
open https://healthcare-rag-app.azurewebsites.net/docs
```

---

## Environment Variables (set securely via Azure, not in code)

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ | LLM generation via Groq API |
| `HF_TOKEN` | ✅ | Download BioBERT + FAISS data from HuggingFace |
| `API_KEY` | Optional | Protects `/query` endpoint (leave empty to disable) |
| `DEPLOY_ENV` | Auto | Set to `azure` by deploy script |
| `WEBSITES_PORT` | Auto | Set by Azure App Service automatically |

---

## Updating After Code Changes

```bash
# Rebuild and push updated image
ACR_NAME="healthcareragacr"
ACR_SERVER=$(az acr show --name $ACR_NAME --query loginServer -o tsv)

docker build -f docker/Dockerfile -t $ACR_SERVER/healthcare-rag:latest .
docker push $ACR_SERVER/healthcare-rag:latest

# Restart the web app to pull latest image
az webapp restart --name healthcare-rag-app --resource-group healthcare-rag-rg
```

---

## Cost Estimate

| Resource | SKU | Monthly Cost |
|---|---|---|
| App Service Plan | B2 (2 vCPU / 3.5 GB) | ~$75 |
| Container Registry | Basic | ~$5 |
| **Total** | | **~$80/month** |

For short-term demo: stop the app when not in use to save costs.

```bash
az webapp stop --name healthcare-rag-app --resource-group healthcare-rag-rg
az webapp start --name healthcare-rag-app --resource-group healthcare-rag-rg
```

---

## Cleanup (Delete Everything)

```bash
az group delete --name healthcare-rag-rg --yes --no-wait
```
