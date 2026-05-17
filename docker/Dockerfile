# Healthcare RAG — Production Dockerfile
# Build:  docker build -f docker/Dockerfile -t healthcare-rag:latest .
# Run:    docker run -p 8000:8000 --env-file .env -v rag-data:/app/data/embeddings --rm healthcare-rag:latest

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Accept HF_TOKEN at build time (used by entrypoint to download models/data)
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first — cached unless requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/      ./src/
COPY api/      ./api/
COPY config/   ./config/
COPY scripts/  ./scripts/
COPY mlops/    ./mlops/
COPY dashboard/ ./dashboard/
COPY setup.py  .
COPY download.py .

# Classifier config only (weights downloaded at runtime from HuggingFace if missing)
COPY models/   ./models/

# Ensure data directories exist (index downloaded at container start if not mounted)
RUN mkdir -p data/embeddings/faiss_index data/processed data/raw

# Install project in editable mode so absolute imports resolve inside Docker
RUN pip install --no-cache-dir -e .

# Entrypoint: downloads index if missing, then starts uvicorn
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000
CMD ["/entrypoint.sh"]
