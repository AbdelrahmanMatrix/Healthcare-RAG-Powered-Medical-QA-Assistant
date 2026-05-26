#!/bin/sh
# entrypoint.sh — Container startup script
#
# Supports both local Docker and Azure App Service:
#   - Azure uses WEBSITES_PORT (plural, with S), set by deploy scripts
#   - Local Docker uses WEBSITE_PORT
#   - Falls back to 8000 if neither is set
#
# NOTE: Data download (FAISS index, CSVs) is handled inside the FastAPI
# lifespan on startup, NOT here. This lets uvicorn bind the port immediately
# so Azure App Service health checks pass instead of timing out with 503.
set -e

# Azure App Service uses WEBSITES_PORT; local Docker uses WEBSITE_PORT
PORT="${WEBSITES_PORT:-${WEBSITE_PORT:-8000}}"

echo "Starting Healthcare RAG API on port ${PORT}..."
exec uvicorn api.main:app \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --workers 1 \
    --log-level info
