#!/bin/sh
# entrypoint.sh — Container startup script
#
# Supports both local Docker and Azure App Service:
#   - Azure uses WEBSITES_PORT (plural, with S), set by deploy scripts
#   - Local Docker uses WEBSITE_PORT
#   - Falls back to 8000 if neither is set
#
# Performance features:
#   - uvloop:     faster asyncio event loop (Linux/macOS only; no-op on Windows)
#   - httptools:  faster HTTP/1.1 request parser
#   Both are installed in requirements.txt and speed up throughput by ~20-40%.
#
# NOTE: Data download (FAISS index, CSVs) and pipeline warm-up happen in
# BACKGROUND TASKS spawned by the FastAPI lifespan, NOT here. The lifespan
# yields immediately so uvicorn serves HTTP requests (including /health probes
# from Azure) while initialization completes asynchronously.
# See api/main.py lifespan() for the background task implementation.
set -e

# Azure App Service uses WEBSITES_PORT; local Docker uses WEBSITE_PORT
PORT="${WEBSITES_PORT:-${WEBSITE_PORT:-8000}}"

# Number of uvicorn workers.
# B2 App Service Plan has 2 vCPUs and 3.5 GB RAM. ML models consume ~1.5 GB,
# leaving ~2 GB for worker overhead. Running 2 workers doubles throughput for
# concurrent requests at the cost of double memory per model copy.
# Override with UVICORN_WORKERS=2 env var when the plan has >= 6 GB RAM.
WORKERS="${UVICORN_WORKERS:-1}"

# Log available memory for debugging OOM issues on constrained plans
echo "========================================================"
echo "  Healthcare RAG API"
echo "  Port:    ${PORT}"
echo "  Workers: ${WORKERS}"
if command -v free >/dev/null 2>&1; then
    echo "  Memory:"
    free -h | grep -v total | head -2
elif [ -r /proc/meminfo ]; then
    echo "  Memory:"
    grep -E "^(MemTotal|MemAvailable|MemFree)" /proc/meminfo
fi
echo "========================================================"

exec uvicorn api.main:app \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --workers "${WORKERS}" \
    --loop uvloop \
    --http httptools \
    --log-level info \
    --no-access-log
