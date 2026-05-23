#!/bin/bash
# entrypoint.sh — Container startup script
#
# Supports both local Docker and Azure App Service:
#   - Azure uses WEBSITES_PORT (plural, with S)
#   - Local Docker uses WEBSITE_PORT
#   - Falls back to 8000 if neither is set
set -e

INDEX_PATH="data/embeddings/faiss_index/pubmedqa_index_flatip.faiss"
MAPPING_PATH="data/embeddings/faiss_index/chunk_mapping.pkl"

if [ ! -f "$INDEX_PATH" ] || [ ! -f "$MAPPING_PATH" ]; then
    echo "⬇️  FAISS index or chunk mapping not found — downloading from HuggingFace..."
    python download.py
    if [ ! -f "$INDEX_PATH" ]; then
        echo "❌ Download failed. Set HF_TOKEN env var and check AbdoMatrix/healthcare-rag-data exists."
        exit 1
    fi
    echo "✅ Data ready."
fi

# Azure App Service uses WEBSITES_PORT; local Docker uses WEBSITE_PORT
PORT="${WEBSITES_PORT:-${WEBSITE_PORT:-8000}}"

echo "🚀 Starting Healthcare RAG API on port ${PORT}..."
exec uvicorn api.main:app \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --workers 1 \
    --log-level info
