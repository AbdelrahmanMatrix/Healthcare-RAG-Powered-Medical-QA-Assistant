#!/bin/sh
# entrypoint.sh — Container startup script
#
# Supports both local Docker and Azure App Service:
#   - Azure uses WEBSITES_PORT (plural, with S), set by deploy scripts
#   - Local Docker uses WEBSITE_PORT
#   - Falls back to 8000 if neither is set
#
# NOTE: Data download (FAISS index, CSVs) and model pre-loading happen
# in a BACKGROUND TASK spawned by the FastAPI lifespan, NOT here.
# The lifespan yields immediately so uvicorn serves HTTP requests (including
# /health probes from Azure) while initialization runs asynchronously.
# See api/main.py lifespan() for the background task implementation.
set -e

# Azure App Service uses WEBSITES_PORT; local Docker uses WEBSITE_PORT
PORT="${WEBSITES_PORT:-${WEBSITE_PORT:-8000}}"

echo "Starting Healthcare RAG API on port ${PORT}..."
exec uvicorn api.main:app \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --workers 1 \
    --log-level info
