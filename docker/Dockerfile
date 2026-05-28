# Healthcare RAG — Production Dockerfile (Multi-Stage)
# Build:  docker build -f docker/Dockerfile -t healthcare-rag:latest .
# Run:    docker run -p 8000:8000 --env-file .env --rm healthcare-rag:latest
#
# ── Design Rationale ──────────────────────────────────────────────────────────
# MULTI-STAGE BUILD:
#   Stage 1 (builder):  Install all Python deps + pre-download HF models into
#                       /app/hf_cache. Build tools (gcc, etc.) stay in this stage.
#   Stage 2 (runtime):  Python-slim base. Copies only installed packages,
#                       the HF cache, and source code. Much smaller image.
#
# HF_TOKEN SECRET:
#   HF_TOKEN is passed via Docker BuildKit secret mount — it is NEVER written
#   to any image layer or docker history. Eliminates the ARG/ENV secret warning.
#   Build with:  DOCKER_BUILDKIT=1 docker build --secret id=HF_TOKEN,env=HF_TOKEN ...
#   (GitHub Actions azure-deploy.yml passes it automatically.)
#
# MODEL PRE-DOWNLOAD STRATEGY:
#   If HF_TOKEN is set:  classifier + embedding + reranker are baked into the
#                        image → cold starts skip the ~2 GB download entirely.
#   If HF_TOKEN is absent (CI smoke builds):  pre_download_models.py exits 0
#                        gracefully → models download lazily at runtime.
#   Classifier not on Hub: non-fatal; pipeline uses local disk weights or the
#                        DistilBERT fallback repo at runtime.
#
#   Data artifacts (FAISS index, CSVs) are NEVER baked in — they download at
#   runtime via the FastAPI lifespan background task (download.py).
# ──────────────────────────────────────────────────────────────────────────────

# syntax=docker/dockerfile:1.4

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1: Builder — install deps + pre-download HF model cache
# ══════════════════════════════════════════════════════════════════════════════
FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Point all HF libraries at a single cache directory
ENV HF_HOME=/app/hf_cache
ENV SENTENCE_TRANSFORMERS_HOME=/app/hf_cache/sentence_transformers
ENV TORCH_HOME=/app/hf_cache/torch
ENV TRANSFORMERS_CACHE=/app/hf_cache/transformers

WORKDIR /app

# Pre-create HF cache directory so the COPY in stage 2 always succeeds
# even when no models were downloaded (CI builds without HF_TOKEN).
RUN mkdir -p /app/hf_cache

# System build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code (needed by pre_download_models.py)
COPY src/      ./src/
COPY api/      ./api/
COPY config/   ./config/
COPY mlops/    ./mlops/
COPY dashboard/ ./dashboard/
COPY setup.py  .
COPY download.py .
COPY pre_download_models.py .
COPY models/   ./models/

# Install project in editable mode
RUN pip install --no-cache-dir -e .

# ── Pre-download HF models using a BuildKit secret ────────────────────────────
# HF_TOKEN is read from the secret mount — never written to any image layer.
# If no secret is provided (CI builds), the script exits 0 and skips downloads.
# Build with: DOCKER_BUILDKIT=1 docker build --secret id=HF_TOKEN,env=HF_TOKEN ...
RUN --mount=type=secret,id=HF_TOKEN,required=false \
    HF_TOKEN=$(cat /run/secrets/HF_TOKEN 2>/dev/null || echo "") \
    python pre_download_models.py

# Clean up caches to reduce layer size
RUN pip cache purge || true && \
    find /root/.cache/pip -type f -delete 2>/dev/null || true && \
    find /tmp -maxdepth 1 -type f -delete 2>/dev/null || true

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2: Runtime — lean image with only what's needed to serve requests
# ══════════════════════════════════════════════════════════════════════════════
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# HF_TOKEN is injected at runtime via Azure App Service app settings or .env.
# It is NOT baked into the image.

# Point to the pre-downloaded (or empty) cache directory
ENV HF_HOME=/app/hf_cache
ENV SENTENCE_TRANSFORMERS_HOME=/app/hf_cache/sentence_transformers
ENV TORCH_HOME=/app/hf_cache/torch
ENV TRANSFORMERS_CACHE=/app/hf_cache/transformers

WORKDIR /app

# Runtime system deps — curl for health probes only; no build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Azure App Service log streaming
RUN mkdir -p /home/LogFiles && ln -sf /dev/stdout /home/LogFiles/entrypoint.log

# ── Copy Python packages ──────────────────────────────────────────────────────
COPY --from=builder /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# ── Copy HF model cache (populated at build time, or empty for CI builds) ─────
COPY --from=builder /app/hf_cache/ /app/hf_cache/

# ── Copy source code ──────────────────────────────────────────────────────────
COPY --from=builder /app/src/      ./src/
COPY --from=builder /app/api/      ./api/
COPY --from=builder /app/config/   ./config/
COPY --from=builder /app/mlops/    ./mlops/
COPY --from=builder /app/dashboard/ ./dashboard/
COPY --from=builder /app/setup.py  .
COPY --from=builder /app/download.py .
COPY --from=builder /app/models/   ./models/

# Ensure data directories exist (artifacts download at runtime)
RUN mkdir -p data/embeddings/faiss_index data/processed data/raw

# Entrypoint: starts uvicorn; data download and pipeline warm-up run
# asynchronously in the background so /health is immediately reachable.
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Docker-native health check — validates the API is responding.
# Azure App Service also has its own HTTP probe (WEBSITES_PORT).
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD curl -sf http://localhost:${WEBSITES_PORT:-8000}/health || exit 1

EXPOSE 8000
CMD ["/entrypoint.sh"]
