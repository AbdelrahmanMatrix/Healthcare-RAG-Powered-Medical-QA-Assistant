#!/bin/bash
set -e

INDEX_PATH="data/embeddings/faiss_index/pubmedqa_index_flatl2.faiss"
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

echo "🚀 Starting Healthcare RAG API on port ${WEBSITE_PORT:-8000}..."
exec uvicorn api.main:app \
    --host 0.0.0.0 \
    --port "${WEBSITE_PORT:-8000}" \
    --workers 1 \
    --log-level info
