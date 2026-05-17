# Healthcare RAG — Production Dockerfile
# Build:  docker build -f docker/Dockerfile -t healthcare-rag:latest .
# Run:    docker run -p 8000:8000 --env-file .env --rm healthcare-rag:latest

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# System deps needed to compile some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first — cached layer rebuilds only when requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/      ./src/
COPY api/      ./api/
COPY config/   ./config/
COPY scripts/  ./scripts/
COPY setup.py  .

# FAISS index + chunk mapping (produced by notebook 05 or python download.py)
COPY data/embeddings/ ./data/embeddings/

# Install project in editable mode so absolute imports work inside Docker
RUN pip install --no-cache-dir -e .

EXPOSE 8000

# Single worker — Azure Free Tier has 1GB RAM limit
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
