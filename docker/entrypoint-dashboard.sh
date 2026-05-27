#!/bin/sh
# entrypoint-dashboard.sh — Container startup script for Streamlit Dashboard
#
# Supports both local Docker and Azure App Service:
#   - Azure uses WEBSITES_PORT (plural, with S), set by deploy scripts
#   - Local Docker uses WEBSITE_PORT
#   - Falls back to 8501 if neither is set
set -e

# Azure App Service uses WEBSITES_PORT; local Docker uses WEBSITE_PORT
PORT="${WEBSITES_PORT:-${WEBSITE_PORT:-8501}}"

echo "Starting Healthcare RAG Dashboard on port ${PORT}..."
exec streamlit run dashboard/app.py \
    --server.port "${PORT}" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.fileWatcherType none \
    --server.enableWebsocketCompression false \
    --runner.fastReruns false \
    --global.developmentMode false \
    --server.maxUploadSize 1 \
    --browser.gatherUsageStats false
