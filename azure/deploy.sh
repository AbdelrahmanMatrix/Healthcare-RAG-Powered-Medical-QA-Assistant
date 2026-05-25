#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Healthcare RAG — Azure App Service Deployment Script
#
# Prerequisites:
#   1. Azure CLI installed:  https://docs.microsoft.com/en-us/cli/azure/install-azure-cli
#   2. Logged in:            az login
#   3. Docker installed and running (for local image build)
#   4. .env file exists with GROQ_API_KEY and HF_TOKEN set
#
# Usage:
#   cd <repo-root>
#   bash azure/deploy.sh
#
# What this script does:
#   1. Creates a Resource Group
#   2. Creates an Azure Container Registry (ACR)
#   3. Builds the Docker image and pushes it to ACR
#   4. Creates an App Service Plan (Linux, B2 — 2 vCPUs / 3.5 GB RAM)
#   5. Creates a Web App for Containers
#   6. Sets all environment variables from your .env file
#   7. Enables continuous deployment from ACR
# ──────────────────────────────────────────────────────────────────────────────
set -e

# ── Configuration — edit these if needed ─────────────────────────────────────
RESOURCE_GROUP="healthcare-rag-rg"
LOCATION="germanywestcentral"
# Resource group already exists here from previous deployment
ACR_NAME="healthcareragacr"          # must be globally unique, lowercase, 5-50 chars
APP_NAME="healthcare-rag-app"        # must be globally unique
APP_SERVICE_PLAN="healthcare-rag-plan"
SKU="B2"                             # B2: 2 vCPUs, 3.5 GB RAM (~$0.10/hr)
                                     # Upgrade to P1v3 for production
# ─────────────────────────────────────────────────────────────────────────────

echo "════════════════════════════════════════════════════════════════"
echo "  Healthcare RAG — Azure Deployment"
echo "════════════════════════════════════════════════════════════════"

# Load .env for secret values
if [ ! -f ".env" ]; then
    echo "❌ .env file not found. Copy .env.example to .env and fill in values."
    exit 1
fi
source .env

if [ -z "$GROQ_API_KEY" ]; then
    echo "❌ GROQ_API_KEY is not set in .env"
    exit 1
fi
if [ -z "$HF_TOKEN" ]; then
    echo "❌ HF_TOKEN is not set in .env"
    exit 1
fi

echo ""
echo "▶ Step 1/7 — Creating Resource Group: $RESOURCE_GROUP in $LOCATION"
az group create \
    --name "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --output table

echo ""
echo "▶ Step 2/7 — Creating Azure Container Registry: $ACR_NAME"
az acr create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$ACR_NAME" \
    --sku Basic \
    --admin-enabled true \
    --output table

# Get ACR login server
ACR_LOGIN_SERVER=$(az acr show --name "$ACR_NAME" --query loginServer --output tsv)
ACR_PASSWORD=$(az acr credential show --name "$ACR_NAME" --query "passwords[0].value" --output tsv)

echo ""
echo "▶ Step 3/7 — Building and pushing Docker image to $ACR_LOGIN_SERVER"
docker build \
    -f docker/Dockerfile \
    -t "${ACR_LOGIN_SERVER}/healthcare-rag:latest" \
    .

docker login "$ACR_LOGIN_SERVER" \
    --username "$ACR_NAME" \
    --password "$ACR_PASSWORD"

docker push "${ACR_LOGIN_SERVER}/healthcare-rag:latest"

echo ""
echo "▶ Step 4/7 — Creating App Service Plan: $APP_SERVICE_PLAN ($SKU)"
az appservice plan create \
    --name "$APP_SERVICE_PLAN" \
    --resource-group "$RESOURCE_GROUP" \
    --is-linux \
    --sku "$SKU" \
    --output table

echo ""
echo "▶ Step 5/7 — Creating Web App: $APP_NAME"
az webapp create \
    --resource-group "$RESOURCE_GROUP" \
    --plan "$APP_SERVICE_PLAN" \
    --name "$APP_NAME" \
    --deployment-container-image-name "${ACR_LOGIN_SERVER}/healthcare-rag:latest" \
    --output table

echo ""
echo "▶ Step 6/7 — Configuring environment variables"
az webapp config appsettings set \
    --resource-group "$RESOURCE_GROUP" \
    --name "$APP_NAME" \
    --settings \
        GROQ_API_KEY="$GROQ_API_KEY" \
        HF_TOKEN="$HF_TOKEN" \
        API_KEY="${API_KEY:-}" \
        DEPLOY_ENV="azure" \
        DEPLOY_DATE="$(date +%Y-%m-%d)" \
        AZURE_APP_URL="https://${APP_NAME}.azurewebsites.net" \
        WEBSITES_PORT="8000" \
    --output table

echo ""
echo "▶ Step 7/7 — Enabling ACR pull + configuring container registry"
az webapp config container set \
    --name "$APP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --docker-custom-image-name "${ACR_LOGIN_SERVER}/healthcare-rag:latest" \
    --docker-registry-server-url "https://${ACR_LOGIN_SERVER}" \
    --docker-registry-server-user "$ACR_NAME" \
    --docker-registry-server-password "$ACR_PASSWORD" \
    --output table

DEPLOY_URL="https://${APP_NAME}.azurewebsites.net"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ✅ Deployment complete!"
echo ""
echo "  API URL:    ${DEPLOY_URL}"
echo "  Health:     ${DEPLOY_URL}/health"
echo "  Swagger:    ${DEPLOY_URL}/docs"
echo "  Query:      POST ${DEPLOY_URL}/query"
echo ""
echo "  ⏳ First startup takes 2–5 minutes (downloading FAISS index)."
echo "  Watch logs: az webapp log tail --name $APP_NAME --resource-group $RESOURCE_GROUP"
echo "════════════════════════════════════════════════════════════════"

# Update AZURE_APP_URL in .env
if grep -q "^AZURE_APP_URL=" .env; then
    sed -i "s|^AZURE_APP_URL=.*|AZURE_APP_URL=${DEPLOY_URL}|" .env
else
    echo "AZURE_APP_URL=${DEPLOY_URL}" >> .env
fi
echo "  .env updated with AZURE_APP_URL=${DEPLOY_URL}"
