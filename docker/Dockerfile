# Healthcare RAG — Production Dockerfile (Multi-Stage)
# Build:  docker build -f docker/Dockerfile -t healthcare-rag:latest .
# Run:    docker run -p 8000:8000 --env-file .env --rm healthcare-rag:latest
#
# ── Design Rationale ──────────────────────────────────────────────────────────
# MULTI-STAGE BUILD:
#   Stage 1 (builder):  Install all Python deps + pre-configure the HF cache
#                       directory. Includes build tools (gcc, etc.) that are
#                       NOT needed at runtime.
#   Stage 2 (runtime):  Python-slim base. Copies ONLY the installed packages,
#                       HF cache directory structure, and source code.
#                       Much smaller image.
#
# HF CACHE STRATEGY:
#   The HF_HOME directory (/app/hf_cache) is created and POPULATED during the
#   BUILD phase by running pre_download_models.py (requires HF_TOKEN build-arg).
#   All models (classifier, embeddings, reranker) are baked into the image,
#   so cold starts skip the ~2 GB download entirely. Data artifacts (FAISS
#   index, CSVs) are still downloaded at runtime on first cold start via
#   the FastAPI lifespan (download.py).
# ──────────────────────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1: Builder — install deps + prepare HF cache directory
# ══════════════════════════════════════════════════════════════════════════════
FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set HF_HOME so we can easily copy the cache to runtime
ENV HF_HOME=/app/hf_cache
ENV SENTENCE_TRANSFORMERS_HOME=/app/hf_cache/sentence_transformers
ENV TORCH_HOME=/app/hf_cache/torch
ENV TRANSFORMERS_CACHE=/app/hf_cache/transformers

WORKDIR /app

# Pre-create HF cache directory so COPY --from=builder in stage 2 succeeds
# even if no models were downloaded during the build (they download at runtime).
RUN mkdir -p /app/hf_cache

# System build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code (needed by download_models.py)
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

# HF_TOKEN must be passed as --build-arg at build time so that
# download_models.py can authenticate with HuggingFace Hub.
# See azure-deploy.yml, build-and-push-api job.
ARG HF_TOKEN
ENV HF_TOKEN=$HF_TOKEN

# ── Pre-download HF models (populated during build) ────────────────────
# Downloads classifier, embedding, and reranker models into /app/hf_cache
# so they are baked into the runtime stage. Requires HF_TOKEN build-arg.
# Data artifacts (FAISS index, CSVs) still download at runtime.
RUN python pre_download_models.py

# Clean up pip cache to save space
RUN pip cache purge || true
RUN find /root/.cache -type f -delete 2>/dev/null || true
RUN find /tmp -type f -delete 2>/dev/null || true

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2: Runtime — lean image with only what's needed
# ══════════════════════════════════════════════════════════════════════════════
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# HF_TOKEN is provided at runtime via Azure App Service app settings or .env file.
# Do NOT bake it into the image for security reasons.

# Point to the cached models location
ENV HF_HOME=/app/hf_cache
ENV SENTENCE_TRANSFORMERS_HOME=/app/hf_cache/sentence_transformers
ENV TORCH_HOME=/app/hf_cache/torch
ENV TRANSFORMERS_CACHE=/app/hf_cache/transformers

WORKDIR /app

# Runtime system deps (minimal — NO build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for Azure App Service log streaming
RUN mkdir -p /home/LogFiles && ln -sf /dev/stdout /home/LogFiles/entrypoint.log

# ── Copy Python packages from builder ────────────────────────────────────────
COPY --from=builder /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# ── Copy HF cache directory (models pre-downloaded) from builder ────────────
# Copies the populated /app/hf_cache directory (classifier, embeddings,
# reranker) to the runtime stage. Cold starts skip the ~2 GB download
# since models are already baked into the image.
COPY --from=builder /app/hf_cache/ /app/hf_cache/

# ── Copy source code ─────────────────────────────────────────────────────────
COPY --from=builder /app/src/      ./src/
COPY --from=builder /app/api/      ./api/
COPY --from=builder /app/config/   ./config/
COPY --from=builder /app/mlops/    ./mlops/
COPY --from=builder /app/dashboard/ ./dashboard/
COPY --from=builder /app/setup.py  .
COPY --from=builder /app/download.py .
COPY --from=builder /app/models/   ./models/

# Ensure data directories exist
RUN mkdir -p data/embeddings/faiss_index data/processed data/raw

# Entrypoint: starts uvicorn (data download skipped if already present)
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000
CMD ["/entrypoint.sh"]
