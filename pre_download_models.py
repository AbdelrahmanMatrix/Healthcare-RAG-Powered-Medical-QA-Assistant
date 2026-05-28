"""
Pre-download all HuggingFace models for the Docker build.

Downloads the classifier, embedding, and reranker models into HF_HOME
(/app/hf_cache) so they are baked into the Docker image and cold starts
skip the ~2 GB download entirely.

This script is intended to run ONLY during the Docker build (builder stage).
In production and local development, models download lazily on first use.

Usage (Dockerfile builder stage):
    ARG HF_TOKEN
    ENV HF_TOKEN=$HF_TOKEN
    RUN python pre_download_models.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

# Model identifiers (must match config/settings.py and src/rag/pipeline.py)
CLASSIFIER_REPO = "AbdoMatrix/biobert-medical-classifier"
EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"


def _format_size(path: str) -> str:
    """Return a human-readable size for the given path."""
    import subprocess
    try:
        result = subprocess.run(
            ["du", "-sh", path],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            return result.stdout.strip().split("\t")[0]
    except Exception:
        pass
    return "(unknown)"


def main():
    print("=" * 60)
    print("Pre-downloading HF models for Docker image build")
    print("=" * 60)

    hf_home = os.environ.get("HF_HOME", "")
    hf_token = os.environ.get("HF_TOKEN", "")
    print(f"  HF_HOME  = {hf_home}")
    print(f"  HF_TOKEN = {'✓ set' if hf_token else '✗ NOT SET'}")
    print()

    if not hf_home:
        print("ERROR: HF_HOME is not set. This script must run in a Docker build "
              "environment where HF_HOME points to /app/hf_cache.")
        sys.exit(1)

    # ── 1. Classifier (BioBERT) ──────────────────────────────────────────
    print("[1/3] Downloading classifier model...")
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    print(f"       Model: {CLASSIFIER_REPO}")
    _ = AutoTokenizer.from_pretrained(CLASSIFIER_REPO, token=hf_token)
    _ = AutoModelForSequenceClassification.from_pretrained(
        CLASSIFIER_REPO, token=hf_token
    )
    print(f"       ✔ Classifier cached ({_format_size(hf_home)})")

    # ── 2. Embedding model (PubMedBERT) ──────────────────────────────────
    print()
    print("[2/3] Downloading embedding model...")
    from sentence_transformers import SentenceTransformer

    print(f"       Model: {EMBEDDING_MODEL}")
    _ = SentenceTransformer(EMBEDDING_MODEL)
    print(f"       ✔ Embedding model cached ({_format_size(hf_home)})")

    # ── 3. Reranker (CrossEncoder) ───────────────────────────────────────
    print()
    print("[3/3] Downloading reranker model...")
    from sentence_transformers import CrossEncoder

    print(f"       Model: {RERANKER_MODEL}")
    _ = CrossEncoder(RERANKER_MODEL)
    print(f"       ✔ Reranker cached ({_format_size(hf_home)})")

    print()
    print("=" * 60)
    print("All models pre-downloaded successfully!")
    print(f"Cache location: {hf_home}")
    print(f"Total size:     {_format_size(hf_home)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
