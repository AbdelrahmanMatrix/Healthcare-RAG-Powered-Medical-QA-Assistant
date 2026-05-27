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

On Windows, run the shell scripts from Git Bash, WSL, or the GitHub Actions
workflow. Plain PowerShell cannot execute the `.sh` scripts directly.

The script:
- Creates Resource Group, Container Registry, App Service Plan, and Web App
- Builds and pushes the Docker image
- Sets all environment variables securely (no secrets in code)
- Prints the live URL when done

**First startup takes 2–5 minutes** — the container downloads the FAISS index, CSVs, and pre-loads models inside the FastAPI lifespan. The port is listening immediately, so Azure health probes do not time out. See `README.md` → [Startup Sequence](../README.md#startup-sequence) for details.

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
| `DEPLOY_DATE` | Auto | Set to deployment date by deploy script |
| `AZURE_APP_URL` | Auto | API endpoint URL set by deploy script |
| `DASHBOARD_URL` | Auto | Dashboard endpoint URL set by deploy script |
| `WEBSITES_PORT` | Auto | `8000` (API) or `8501` (Dashboard) for App Service routing |
| `WEBSITES_CONTAINER_START_TIME_LIMIT` | Auto | Set to `1800` (30 min) for cold start tolerance |

---

## Updating After Code Changes

### API App Service

```bash
ACR_NAME="healthcareragacr"
RESOURCE_GROUP="healthcare-rag-rg"
SERVER="${ACR_NAME}.azurecr.io"

# Build and push
az acr login --name "$ACR_NAME"
docker build -f docker/Dockerfile -t "${SERVER}/healthcare-rag:latest" .
docker push "${SERVER}/healthcare-rag:latest"

# Restart to pull latest image
az webapp restart --name healthcare-rag-app --resource-group "$RESOURCE_GROUP"
```

### Dashboard App Service

```bash
ACR_NAME="healthcareragacr"
RESOURCE_GROUP="healthcare-rag-rg"
SERVER="${ACR_NAME}.azurecr.io"

# Build and push
az acr login --name "$ACR_NAME"
docker build -f docker/Dockerfile.dashboard -t "${SERVER}/healthcare-rag-dashboard:latest" .
docker push "${SERVER}/healthcare-rag-dashboard:latest"

# Restart to pull latest image
az webapp restart --name healthcare-rag-dashboard --resource-group "$RESOURCE_GROUP"
```

> **Note:** The server URL is computed inline as `${ACR_NAME}.azurecr.io` rather than
> fetched via `az acr show`. This avoids masked-secret issues in CI/CD environments
> where the ACR login server value may trigger secret masking.

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
