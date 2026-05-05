# Milestone 3 — Complete Step-by-Step Guide

**Healthcare RAG Medical Q&A Assistant | eyouth × DEPI | Deadline: May 25, 2026**

---

## Before You Start — Read This

Your M2 code is complete and working. M3 adds a FastAPI layer on top of it, packages everything into Docker, and deploys it to Azure. You are NOT rewriting M2.

**What is already done (do not touch these):**
- `src/rag/pipeline.py` — complete
- `src/classification/classifier.py` — complete
- All `__init__.py` files — already exist
- `setup.py`, `.gitignore` — complete

**What needs to happen in M3:**
1. Fix 3 existing files (small edits)
2. Write 9 empty files (api + docker)
3. Create 5 missing files (scripts + reports + infra)
4. Upload classifier weights to HuggingFace (blocker for everyone)
5. Deploy to Azure

---

## Step 1 — Abdelrahman: Upload the Classifier to HuggingFace (DO THIS FIRST)

**Why this must be first:** The Docker container downloads the DistilBERT classifier from `huggingface.co/AbdooMatrix/distilbert-medical-classifier` when it starts in Azure. That repo does not exist yet. Until it does, nothing can be deployed.

```bash
# Install the HuggingFace upload library
pip install huggingface_hub

# Log in (get your token from https://huggingface.co/settings/tokens)
huggingface-cli login
```

Create the file `scripts/upload_classifier_to_hub.py`:

```python
"""
Run once to upload trained DistilBERT weights to HuggingFace Hub.
After this, the Docker container can download them in Azure.
"""
from pathlib import Path
from huggingface_hub import HfApi, create_repo

HF_REPO_ID = "AbdooMatrix/distilbert-medical-classifier"
LOCAL_PATH  = Path("models/classifier/distilbert_classifier")

def main():
    # Validate local weights exist
    if not LOCAL_PATH.exists():
        raise FileNotFoundError(
            f"{LOCAL_PATH} not found. Run notebook 04_classification.ipynb first."
        )
    has_weights = any(
        (LOCAL_PATH / f).exists()
        for f in ["model.safetensors", "pytorch_model.bin"]
    )
    if not has_weights:
        raise FileNotFoundError(
            f"No .safetensors or .bin file in {LOCAL_PATH}. Train the model first."
        )
    print(f"Local weights found in {LOCAL_PATH}:")
    for f in LOCAL_PATH.iterdir():
        print(f"  {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)")

    # Create public repo
    create_repo(repo_id=HF_REPO_ID, repo_type="model", exist_ok=True, private=False)
    print(f"\nRepo ready: https://huggingface.co/{HF_REPO_ID}")

    # Upload all files
    HfApi().upload_folder(
        folder_path=str(LOCAL_PATH),
        repo_id=HF_REPO_ID,
        repo_type="model",
        ignore_patterns=["*.pyc", "__pycache__", "checkpoints/"],
    )
    print(f"\nDone. Open https://huggingface.co/{HF_REPO_ID}")
    print("Confirm it is PUBLIC before telling Doha to build Docker.")

if __name__ == "__main__":
    main()
```

Then run it:

```bash
python scripts/upload_classifier_to_hub.py
```

Open `https://huggingface.co/AbdooMatrix/distilbert-medical-classifier` in a browser. You should see `config.json`, `model.safetensors`, `tokenizer_config.json`, `vocab.txt`. **Tell Doha when this is confirmed.**

---

## Step 2 — Everyone: Get the FAISS Index

The FAISS index and data files are gitignored. Run this from the project root to download them from HuggingFace:

```bash
python download.py
```

Expected output: `✅ All N data files already exist.` or it downloads missing files. After this, `data/embeddings/faiss_index/pubmedqa_index_flatl2.faiss` and `data/embeddings/faiss_index/chunk_mapping.pkl` must exist.

---

## Step 3 — Everyone: Install Missing Packages

These 5 packages are missing from `requirements.txt`:

```bash
pip install httpx==0.27.0 pytest-asyncio==0.23.6 azure-storage-blob==12.19.0 \
            requests==2.31.0 flake8==7.0.0 torch==2.2.2
```

Then add them to `requirements.txt`. Open the file and add this section at the bottom:

```text
# ─── M3 additions ─────────────────────────────────────────────────────────
httpx==0.27.0
pytest-asyncio==0.23.6
azure-storage-blob==12.19.0
requests==2.31.0
flake8==7.0.0
torch==2.2.2
```

> **Why torch?** It is currently commented out in `requirements.txt`. The Dockerfile runs `pip install -r requirements.txt` and will skip a commented line, so the classifier crashes inside Docker. Add it as a real line.

---

## Step 4 — Everyone: Fix `config/settings.py`

Open `config/settings.py`. Find the line `GROQ_API_KEY: str = ""`. Add these two fields **directly below it**, before the `class Config:` block:

```python
    # ── M3 additions ──────────────────────────────────────────────────
    disclaimer: str = (
        "This is an informational assistant only. "
        "Always consult a qualified healthcare professional for medical decisions. "
        "Do not use this information as a substitute for professional medical advice, "
        "diagnosis, or treatment."
    )

    categories: list = [
        "Symptoms", "Diagnosis", "Treatment",
        "Medication", "Prevention", "General"
    ]
```

Do not change anything else in this file.

---

## Step 5 — Everyone: Fix `src/pipeline.py`

The current file has a Python bug (`global _rag` + `try/except NameError` in the same scope — does not work). Replace the **entire file** with this:

```python
"""
Top-level pipeline entry point.
- run_query()    → kept for notebook/M2 test compatibility
- run_pipeline() → called by the FastAPI /query endpoint
"""
from src.classification.classifier import predict
from src.rag.pipeline import build_rag_pipeline

_rag = None  # module-level singleton, loaded once on first request


def _get_rag():
    global _rag
    if _rag is None:
        _rag = build_rag_pipeline()
    return _rag


def run_query(query: str) -> dict:
    """Original pipeline — returns full dict including disclaimer baked into 'answer'."""
    category = predict(query)
    return _get_rag().answer_with_routing(query, category=category)


def run_pipeline(question: str) -> dict:
    """
    M3 API entry point. Called by api/routes/query.py.
    Returns answer WITHOUT disclaimer (the API layer injects it separately).
    Returns source IDs as plain strings (the API schema requires List[str]).
    """
    result = run_query(question)
    raw_sources = result.get("retrieved_sources", [])
    source_ids  = [str(s["chunk_id"]) for s in raw_sources] if raw_sources else []
    return {
        "answer":   result.get("answer_raw", result.get("answer", "")),
        "category": result.get("category", "General"),
        "sources":  source_ids,
    }
```

---

## Step 6 — Everyone: Fix `tests/conftest.py`

Open `tests/conftest.py`. Add the following **at the very top of the file**, before anything else that is already there. Do not remove the existing `repo_root` and `sample_queries` fixtures:

```python
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from api.main import app

MOCK_RESULT = {"answer": "Mock answer.", "category": "Symptoms", "sources": ["42"]}


@pytest.fixture
def api_client():
    return TestClient(app)


@pytest.fixture
def mock_pipeline():
    with patch("api.routes.query.run_pipeline") as mock:
        mock.return_value = MOCK_RESULT
        yield mock


@pytest.fixture
def mock_pipeline_error():
    with patch("api.routes.query.run_pipeline") as mock:
        mock.side_effect = RuntimeError("Simulated failure")
        yield mock


@pytest.fixture
def sample_question():
    return "What are the main symptoms of type 2 diabetes?"


# ── existing fixtures below — keep them as-is ────────────────────────────
```

---

## Step 7 — Youssef: Write the FastAPI Files (Task 1, Days 1–5)

All 6 files below are 0 bytes in the repo. Write them in the order shown.

---

### `api/__init__.py`

```python
# api package
```

---

### `api/schemas/__init__.py`

```python
# api.schemas package
```

---

### `api/schemas/request.py`

```python
from pydantic import BaseModel, Field
from typing import List, Optional


class QueryRequest(BaseModel):
    question: str = Field(
        ..., min_length=5, max_length=1000,
        example="What are the symptoms of type 2 diabetes?"
    )


class QueryResponse(BaseModel):
    answer:            str
    category:          str
    retrieved_sources: List[str]
    disclaimer:        str   # always present — injected from config.settings


class HealthResponse(BaseModel):
    status:       str            = "ok"
    model_loaded: Optional[bool] = None
```

---

### `api/routes/__init__.py`

> ⚠️ This file must use a **relative** import. Writing `from api.routes import query` causes a circular import error and the app will not start.

```python
# api.routes package
from . import query   # relative import — do not change this to absolute
__all__ = ["query"]
```

---

### `api/routes/query.py`

```python
import logging
from fastapi import APIRouter, HTTPException
from api.schemas.request import QueryRequest, QueryResponse, HealthResponse
from config.settings import settings
from src.pipeline import run_pipeline

logger = logging.getLogger("healthcare_rag.routes")
router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest) -> QueryResponse:
    """Run the RAG pipeline and return a grounded answer with mandatory disclaimer."""
    logger.info(f"Query: {request.question[:80]}")
    try:
        result = run_pipeline(request.question)
        return QueryResponse(
            answer            = result.get("answer", "Unable to generate a response."),
            category          = result.get("category", "General"),
            retrieved_sources = result.get("sources", []),
            disclaimer        = settings.disclaimer,
        )
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health probe for Azure App Service. Returns 200 instantly, never loads models."""
    return HealthResponse(status="ok", model_loaded=True)
```

---

### `api/main.py`

```python
import time, logging, sys
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from api.routes import query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("healthcare_rag")

app = FastAPI(
    title="Healthcare RAG Medical Q&A API",
    version="1.0.0",
    description="RAG-powered medical Q&A. Every /query response includes a mandatory disclaimer.",
)


@app.middleware("http")
async def latency_middleware(request: Request, call_next):
    """Measures request time and injects X-Response-Time-Ms header into every response."""
    start    = time.perf_counter()
    response = await call_next(request)
    ms       = round((time.perf_counter() - start) * 1000, 2)
    response.headers["X-Response-Time-Ms"] = str(ms)
    logger.info(f"{request.method} {request.url.path} → {response.status_code} | {ms}ms")
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"error": str(exc)})


app.include_router(query.router)


@app.get("/", include_in_schema=False)
async def root():
    return {"project": "Healthcare RAG", "docs": "/docs", "health": "/health"}


@app.on_event("startup")
async def startup():
    logger.info("API started. Swagger UI at /docs")
```

---

### `tests/test_api.py`

```python
"""
M3 Task 1 — API tests. Run: pytest tests/test_api.py -v
All 9 must pass before moving to Docker.
"""
from unittest.mock import patch
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)
MOCK   = {"answer": "Mock answer.", "category": "Symptoms", "sources": ["42"]}


class TestHealth:
    def test_returns_200(self):
        assert client.get("/health").status_code == 200

    def test_status_field_is_ok(self):
        assert client.get("/health").json()["status"] == "ok"

    def test_latency_header_present(self):
        assert "X-Response-Time-Ms" in client.get("/health").headers


class TestQuery:
    @patch("api.routes.query.run_pipeline")
    def test_returns_200(self, m):
        m.return_value = MOCK
        assert client.post("/query", json={"question": "What causes diabetes?"}).status_code == 200

    @patch("api.routes.query.run_pipeline")
    def test_all_required_fields_present(self, m):
        m.return_value = MOCK
        data = client.post("/query", json={"question": "How is hypertension treated?"}).json()
        for field in ("answer", "category", "retrieved_sources", "disclaimer"):
            assert field in data, f"Missing field: {field}"

    @patch("api.routes.query.run_pipeline")
    def test_disclaimer_is_non_empty_string(self, m):
        m.return_value = MOCK
        data = client.post("/query", json={"question": "What is metformin used for?"}).json()
        assert isinstance(data["disclaimer"], str) and len(data["disclaimer"]) > 20

    @patch("api.routes.query.run_pipeline")
    def test_category_is_valid(self, m):
        m.return_value = MOCK
        data = client.post("/query", json={"question": "How to prevent stroke?"}).json()
        assert data["category"] in {
            "Symptoms", "Diagnosis", "Treatment", "Medication", "Prevention", "General"
        }

    @patch("api.routes.query.run_pipeline")
    def test_latency_header_present_on_query(self, m):
        m.return_value = MOCK
        r = client.post("/query", json={"question": "What are flu symptoms?"})
        assert "X-Response-Time-Ms" in r.headers

    def test_missing_question_returns_422(self):
        assert client.post("/query", json={}).status_code == 422

    @patch("api.routes.query.run_pipeline")
    def test_pipeline_error_returns_500(self, m):
        m.side_effect = RuntimeError("FAISS index not found")
        assert client.post("/query", json={"question": "What causes anaemia?"}).status_code == 500
```

**Verify Task 7 is done:**

```bash
pytest tests/test_api.py -v
# Expected: 9 passed

uvicorn api.main:app --reload --port 8000
# Open http://localhost:8000/docs in browser
# Try /query with {"question": "What are symptoms of diabetes?"}
# Confirm the response contains a "disclaimer" field
```

**Commit:**

```bash
git add api/ tests/test_api.py tests/conftest.py src/pipeline.py \
        config/settings.py requirements.txt scripts/upload_classifier_to_hub.py
git commit -m "M3 T1: FastAPI app, schemas, routes, 9 passing tests"
```

---

## Step 8 — Create `Makefile` (project root)

```makefile
APP_NAME  := healthcare-rag
API_PORT  := 8000
ACR_NAME  := healthcareragacr
.DEFAULT_GOAL := help

.PHONY: install test test-api run lint docker-build docker-run docker-push latency-test clean help

install:       ## Install all dependencies
	pip install --upgrade pip && pip install -r requirements.txt && pip install -e .

test:          ## Run all pytest tests
	pytest tests/ -v --tb=short

test-api:      ## Run API tests only (fast, no models needed)
	pytest tests/test_api.py -v

run:           ## Start FastAPI with hot-reload
	uvicorn api.main:app --reload --host 0.0.0.0 --port $(API_PORT)

lint:          ## Run flake8 style check
	flake8 src/ api/ scripts/ --max-line-length 120 --exclude __pycache__

docker-build:  ## Build Docker image
	docker build -f docker/Dockerfile -t $(APP_NAME):latest .

docker-run:    ## Run Docker container locally
	docker run -p $(API_PORT):$(API_PORT) --env-file .env --rm $(APP_NAME):latest

docker-push:   ## Push image to Azure Container Registry
	docker tag $(APP_NAME):latest $(ACR_NAME).azurecr.io/$(APP_NAME):v1
	docker push $(ACR_NAME).azurecr.io/$(APP_NAME):v1

latency-test:  ## Run 20-query latency test against live Azure API
	python scripts/latency_test.py

clean:         ## Remove Python cache files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

help:          ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'
```

---

## Step 9 — Create `.env.example` (project root)

```env
# Copy this file to .env and fill in your values.
# .env is gitignored — never commit it.

WEBSITE_PORT=8000

# HuggingFace token — needed to download the DistilBERT classifier in Docker/Azure
# Get yours at: https://huggingface.co/settings/tokens
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Azure deployment URL — fill in after Step 12 (Azure deployment) is done
AZURE_APP_URL=https://healthcare-rag-app.azurewebsites.net
```

Then create your local `.env`:

```bash
cp .env.example .env
# Edit .env and set your real HF_TOKEN
```

---

## Step 10 — Doha: Write Docker Files (Task 2, Days 4–9)

Both files are 0 bytes. Write them now.

> ⚠️ Do not start this step until Abdelrahman confirms the HuggingFace repo is public (Step 1). The Docker image downloads the classifier from there.

---

### `docker/Dockerfile`

```dockerfile
# Build:  docker build -f docker/Dockerfile -t healthcare-rag:latest .
# Run:    docker run -p 8000:8000 --env-file .env --rm healthcare-rag:latest

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cached layer — only rebuilds when requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/    ./src/
COPY api/    ./api/
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY setup.py .

# Copy FAISS index and data (generated by python download.py or notebook 05)
COPY data/embeddings/ ./data/embeddings/

# Install project so absolute imports (from src.rag.pipeline import ...) work inside Docker
RUN pip install --no-cache-dir -e .

EXPOSE 8000

# Single worker — Azure Free Tier has 1GB RAM limit
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

> **Note on classifier weights:** The classifier is NOT copied into the image. When the container starts in Azure and the first `/query` request arrives, `classifier.py` automatically downloads the weights from `huggingface.co/AbdooMatrix/distilbert-medical-classifier` using the `HF_TOKEN` environment variable. This is why Step 1 (uploading the weights to HuggingFace) must happen first.

---

### `docker/docker-compose.yml`

```yaml
version: "3.9"
services:
  healthcare-rag:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    container_name: healthcare-rag-api
    ports:
      - "8000:8000"
    env_file:
      - ../.env
    restart: on-failure
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 90s
```

---

### `.dockerignore` (project root)

```
.git/
.env
.env.*
notebooks/
mlruns/
reports/
__pycache__/
*.pyc
*.pyo
.ipynb_checkpoints/
.idea/
.vscode/
.pytest_cache/
data/raw/
data/processed/
models/
```

**Build and test locally:**

```bash
make docker-build

# Run the container (flan-t5 loads on first /query call — takes ~60s)
make docker-run

# In a second terminal:
curl http://localhost:8000/health
# Expected: {"status":"ok","model_loaded":true}

curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What are the symptoms of diabetes?"}'
# Expected: JSON with answer, category, retrieved_sources, disclaimer
```

**Commit:**

```bash
git add docker/ .dockerignore Makefile .env.example
git commit -m "M3 T2: Dockerfile, docker-compose, dockerignore, Makefile"
```

---

## Step 11 — Create `scripts/latency_test.py`

```python
"""
M3 Task 4 — 20-query latency test against the live Azure API.
Run: python scripts/latency_test.py
"""
import requests, time, csv, os
from datetime import datetime

BASE_URL   = os.getenv("AZURE_APP_URL", "https://healthcare-rag-app.azurewebsites.net")
VALID_CATS = {"Symptoms", "Diagnosis", "Treatment", "Medication", "Prevention", "General"}
LIMIT_MS   = 5000

TEST_QUERIES = [
    ("What are the early symptoms of type 2 diabetes?",        "Symptoms"),
    ("What symptoms indicate a possible heart attack?",         "Symptoms"),
    ("What are warning signs of kidney disease?",               "Symptoms"),
    ("What symptoms are linked to iron deficiency anaemia?",    "Symptoms"),
    ("How is hypertension diagnosed?",                          "Diagnosis"),
    ("What tests diagnose thyroid disorders?",                  "Diagnosis"),
    ("How do doctors diagnose celiac disease?",                 "Diagnosis"),
    ("What treatments are available for moderate asthma?",      "Treatment"),
    ("How is Crohn's disease treated in adults?",               "Treatment"),
    ("What is the treatment for mild depression?",              "Treatment"),
    ("What are the common side effects of metformin?",          "Medication"),
    ("Is it safe to take ibuprofen and paracetamol together?",  "Medication"),
    ("What are the main uses of corticosteroids?",              "Medication"),
    ("How can I reduce the risk of cardiovascular disease?",    "Prevention"),
    ("What lifestyle changes help prevent type 2 diabetes?",    "Prevention"),
    ("How effective are vaccines in preventing influenza?",      "Prevention"),
    ("What is the difference between a virus and a bacterium?", "General"),
    ("How does the immune system respond to infection?",        "General"),
    ("What is BMI and how is it calculated?",                   "General"),
    ("What does peer-reviewed medical research mean?",          "General"),
]


def warm_up():
    print(f"Target: {BASE_URL}")
    print("Sending warm-up request (not recorded in results)...")
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=120)
        print(f"  Warm-up: HTTP {r.status_code} in {r.elapsed.total_seconds():.1f}s")
    except Exception as e:
        print(f"  Warm-up failed: {e}")
        print("  If Azure was idle, wait 90s and run again.")


def query_once(question: str, n: int) -> dict:
    try:
        t0 = time.perf_counter()
        r  = requests.post(f"{BASE_URL}/query", json={"question": question}, timeout=30)
        ms = round((time.perf_counter() - t0) * 1000, 1)
        if r.status_code == 200:
            d = r.json()
            return {
                "n": n, "question": question, "ms": ms,
                "category": d.get("category", "MISSING"),
                "disc_ok":  bool(d.get("disclaimer")),
                "lat_pass": ms <= LIMIT_MS,
                "cat_valid": d.get("category") in VALID_CATS,
                "error": None,
            }
        return {"n": n, "question": question, "ms": ms, "category": "ERR",
                "disc_ok": False, "lat_pass": False, "cat_valid": False,
                "error": f"HTTP {r.status_code}"}
    except requests.exceptions.Timeout:
        return {"n": n, "question": question, "ms": 30000, "category": "TIMEOUT",
                "disc_ok": False, "lat_pass": False, "cat_valid": False,
                "error": "Timeout after 30s"}


if __name__ == "__main__":
    warm_up()
    print("Waiting 5s for warm instance..."); time.sleep(5)

    results = []
    for i, (q, expected) in enumerate(TEST_QUERIES, 1):
        r = query_once(q, i)
        r["expected"] = expected
        results.append(r)
        icon = "✅" if r["lat_pass"] and r["disc_ok"] else "❌"
        print(f"  [{i:02d}/20] {icon} {r['ms']:>7.1f}ms | {r['category']:<12} | {q[:50]}")
        time.sleep(0.5)

    lat_pass  = sum(1 for r in results if r["lat_pass"])
    disc_pass = sum(1 for r in results if r["disc_ok"])
    avg_ms    = sum(r["ms"] for r in results) / len(results)
    max_ms    = max(r["ms"] for r in results)

    print(f"""
{'='*55}
Latency ≤5000ms : {lat_pass}/20  {'✅ PASS' if lat_pass == 20 else '❌ FAIL'}
Disclaimer OK   : {disc_pass}/20  {'✅ PASS' if disc_pass == 20 else '❌ FAIL'}
Average latency : {avg_ms:.0f} ms
Max latency     : {max_ms:.0f} ms
Run date        : {datetime.now().strftime('%Y-%m-%d %H:%M')}
{'='*55}""")

    os.makedirs("reports", exist_ok=True)
    with open("reports/latency_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "n", "question", "expected", "category",
            "ms", "disc_ok", "lat_pass", "cat_valid", "error"
        ])
        writer.writeheader()
        writer.writerows(results)
    print("Saved: reports/latency_results.csv")
```

---

## Step 12 — Eman: Deploy to Azure (Task 3, Days 7–14)

Run these commands in order. Every one must succeed before running the next.

> **Important:** The env variable is `HF_TOKEN` — NOT `GROQ_API_KEY`. The pipeline uses `google/flan-t5-base` via HuggingFace. Groq is not used anywhere in the actual pipeline code.

```bash
# 1. Log in to Azure
az login
az account show

# 2. Create resource group
az group create --name healthcare-rag-rg --location eastus

# 3. Create Container Registry
az acr create \
  --name           healthcareragacr \
  --resource-group healthcare-rag-rg \
  --sku            Basic \
  --admin-enabled  true

# 4. Push image to ACR
az acr login --name healthcareragacr
docker tag healthcare-rag:latest healthcareragacr.azurecr.io/healthcare-rag:v1
docker push healthcareragacr.azurecr.io/healthcare-rag:v1

# Verify image is there
az acr repository list --name healthcareragacr --output table

# 5. Create App Service Plan (Free Tier)
az appservice plan create \
  --name           healthcare-rag-plan \
  --resource-group healthcare-rag-rg \
  --is-linux --sku F1

# 6. Create Web App
az webapp create \
  --name                            healthcare-rag-app \
  --resource-group                  healthcare-rag-rg \
  --plan                            healthcare-rag-plan \
  --deployment-container-image-name healthcareragacr.azurecr.io/healthcare-rag:v1

# 7. Connect ACR credentials to the Web App
ACR_PASSWORD=$(az acr credential show \
  --name healthcareragacr --query passwords[0].value -o tsv)

az webapp config container set \
  --name                            healthcare-rag-app \
  --resource-group                  healthcare-rag-rg \
  --docker-registry-server-url      https://healthcareragacr.azurecr.io \
  --docker-registry-server-user     healthcareragacr \
  --docker-registry-server-password $ACR_PASSWORD

# 8. Set environment variables
az webapp config appsettings set \
  --name           healthcare-rag-app \
  --resource-group healthcare-rag-rg \
  --settings \
    WEBSITE_PORT="8000" \
    PYTHONUNBUFFERED="1" \
    HF_TOKEN="paste-your-hf-token-here"

# 9. Wait 2–3 minutes, then verify
curl https://healthcare-rag-app.azurewebsites.net/health
# Expected: {"status":"ok","model_loaded":true}

# 10. Take an Azure portal screenshot for KPI 4
# Open https://portal.azure.com in your browser
# Navigate to: Resource Groups → healthcare-rag-rg → healthcare-rag-app
# Take a screenshot showing the app status as "Running"
# Save it as reports/azure_portal_screenshot.png then commit:
mkdir -p reports
git add reports/azure_portal_screenshot.png
git commit -m "M3 T3: Azure portal screenshot for KPI 4"

# 11. View live container logs (useful for debugging)
az webapp log tail \
  --name healthcare-rag-app \
  --resource-group healthcare-rag-rg
```

**As soon as the URL responds with 200:**
- Tell Abdelrahman the URL so he can run the latency test (Step 13)
- Tell Ziad the URL so he can update the proposal (Step 14)

**If you need to redeploy after a code change:**

```bash
docker build -f docker/Dockerfile -t healthcareragacr.azurecr.io/healthcare-rag:v2 .
docker push healthcareragacr.azurecr.io/healthcare-rag:v2
az webapp config container set \
  --name healthcare-rag-app \
  --resource-group healthcare-rag-rg \
  --docker-custom-image-name healthcareragacr.azurecr.io/healthcare-rag:v2
az webapp restart \
  --name healthcare-rag-app \
  --resource-group healthcare-rag-rg
```

> **Cold-start is normal.** Azure Free Tier shuts down after 20 minutes of inactivity. The first request after idle takes 60–120 seconds while flan-t5 reloads. This is excluded from the KPI measurement. Only warm-instance latency counts.

---

## Step 13 — Abdelrahman: Run Latency Test (Task 4, Days 12–19)

```bash
# Set the Azure URL (or edit the default in the script)
export AZURE_APP_URL=https://healthcare-rag-app.azurewebsites.net

python scripts/latency_test.py
```

Expected output:
```
Latency ≤5000ms : 20/20  ✅ PASS
Disclaimer OK   : 20/20  ✅ PASS
Average latency : ___ms
Max latency     : ___ms
```

Then create `reports/deployment_test_report.md` and fill in the results:

```markdown
# M3 Deployment Test Report
**Owner:** Abdelrahman Mostafa Sayed
**Date:** YYYY-MM-DD
**API URL:** https://healthcare-rag-app.azurewebsites.net

## Results

| # | Expected | Actual | Latency (ms) | Disclaimer | Pass |
|---|---|---|---|---|---|
| 01 | Symptoms   | ___ | ___ | ✓ | ✅ |
...fill all 20 rows from script output...
| 20 | General    | ___ | ___ | ✓ | ✅ |

## KPI Summary

| KPI | Target | Result | Status |
|---|---|---|---|
| Warm latency ≤5000ms | 20/20 | ___/20 | ✅/❌ |
| Disclaimer present   | 20/20 | ___/20 | ✅/❌ |
| Average latency      | —     | ___ ms | — |
| Max latency          | ≤5000 | ___ ms | ✅/❌ |
```

**Commit:**

```bash
git add scripts/latency_test.py reports/deployment_test_report.md reports/latency_results.csv
git commit -m "M3 T4: latency test 20/20 pass"
```

---

## Step 14 — Ziad: Write Documentation & Submit (Task 5, Days 17–21)

Create `reports/integration_doc.md`:

```markdown
# M3 Integration Documentation
**Owner:** Ziad Ahmed El-Nady | eyouth × DEPI 2025

## Live URL
https://healthcare-rag-app.azurewebsites.net

## Endpoints
| Endpoint | Method | Returns |
|---|---|---|
| /query  | POST | {answer, category, retrieved_sources, disclaimer} |
| /health | GET  | {status: "ok", model_loaded: true} |
| /docs   | GET  | Swagger UI |

## Architecture
Client → FastAPI (api/main.py)
       → run_pipeline (src/pipeline.py)
           → DistilBERT classify  (src/classification/classifier.py)
           → FAISS retrieve top-5 (src/rag/pipeline.py)
           → flan-t5-base generate (src/rag/pipeline.py)
       → disclaimer injected from config.settings.disclaimer
       → JSON response
       → Docker → Azure Container Registry → Azure App Service F1

## Environment Variables in Azure
| Variable | Value | Purpose |
|---|---|---|
| WEBSITE_PORT | 8000 | Port FastAPI listens on |
| PYTHONUNBUFFERED | 1 | Real-time logs |
| HF_TOKEN | hf_xxx | Downloads DistilBERT classifier weights |

## Azure Limitations (Free Tier F1)
- 1GB RAM, 60 CPU min/day
- Idles after 20 min → cold-start 60-120s (excluded from KPI)
- URL: https://healthcare-rag-app.azurewebsites.net

## Keep this URL live until July 2026 (needed for M4, M5, final presentation)
```

Also open the proposal DOCX and add the live URL to **Section 12**.

**Final commit and push:**

```bash
git add reports/integration_doc.md
git commit -m "M3 T5: integration docs complete"
git push origin main
```

---

## Step 15 — Final Check Before Submitting

Run every command. All must pass.

```bash
# Imports work
python -c "from config.settings import settings; print(settings.disclaimer[:40])"
python -c "from src.pipeline import run_pipeline; print('OK')"
python -c "from api.main import app; print('FastAPI OK')"

# 9 tests pass
pytest tests/test_api.py -v

# No lint errors
make lint

# Docker builds
make docker-build

# Azure URL responds
curl https://healthcare-rag-app.azurewebsites.net/health

# Latency test passes
python scripts/latency_test.py

# No secrets committed
git status   # must NOT show .env, *.safetensors, *.faiss, *.pkl

# Push
git push origin main
```

---

## Who Does What — Summary

| Person | Steps |
|---|---|
| **Abdelrahman** | Step 1 (upload classifier — do first) → Step 13 (latency test) |
| **Everyone** | Steps 2, 3, 4, 5, 6 (setup + small file fixes) |
| **Youssef** | Step 7 (all 6 api/ files + tests) |
| **Eman/Doha** | Step 8, 9 (Makefile, .env.example) |
| **Doha** | Step 10 (Dockerfile + docker-compose) |
| **Eman** | Step 12 (Azure deployment) |
| **Ziad** | Step 14 (integration doc + proposal update + final push) |

---

## KPI Checklist

| # | KPI | How to verify |
|---|---|---|
| 1 | Live Azure URL returns HTTP 200 | `curl https://healthcare-rag-app.azurewebsites.net/health` |
| 2 | All 20 queries return a disclaimer | `python scripts/latency_test.py` → Disclaimer OK: 20/20 |
| 3 | Warm latency ≤ 5000ms for all 20 | `python scripts/latency_test.py` → Latency: 20/20 |
| 4 | Stable Docker deployment on Azure | Azure portal screenshot in `reports/` |
| 5 | API uptime ≥ 90% over 48h | Azure App Service metrics dashboard |

---

*Healthcare RAG — Milestone 3 | eyouth × DEPI 2025 | Deadline: May 25, 2026*
