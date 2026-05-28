"""
Pre-download all HuggingFace models for the Docker build.

Downloads the classifier, embedding, and reranker models into HF_HOME
(/app/hf_cache) so they are baked into the Docker image and cold starts
skip the ~2 GB download entirely.

This script is intended to run ONLY during the Docker build (builder stage).
In production and local development, models download lazily on first use.

Usage (Dockerfile builder stage):
    RUN --mount=type=secret,id=HF_TOKEN,required=false \
        HF_TOKEN=$(cat /run/secrets/HF_TOKEN 2>/dev/null || echo "") \
        python pre_download_models.py

Behaviour matrix:
  HF_TOKEN set + repo exists  → downloads & caches model        ✔
  HF_TOKEN set + repo missing → warns and skips (runtime fallback used)
  HF_TOKEN not set            → skips all downloads gracefully (CI builds)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

# ── Model identifiers (must match config/settings.py and src/rag/pipeline.py) ─────────────────────────────────────────────────
CLASSIFIER_REPO = "AbdoMatrix/biobert-medical-classifier"
CLASSIFIER_FALLBACK_REPO = "AbdoMatrix/distilbert-medical-classifier"
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
    hf_token = os.environ.get("HF_TOKEN", "").strip()

    print(f"  HF_HOME  = {hf_home or '(not set)'}")
    hf_token_status = "\u2713 set" if hf_token else "\u2717 not set"
    print(f"  HF_TOKEN = {hf_token_status}")
    print()

    # ── Guard: HF_HOME must be set (Docker build sets it; local dev does not) ─────────────────────────
    if not hf_home:
        print("ERROR: HF_HOME is not set.")
        print("  This script must run inside a Docker build where HF_HOME=/app/hf_cache.")
        sys.exit(1)

    # ── Guard: skip all downloads when HF_TOKEN is absent (CI smoke builds) ───────────────────
    if not hf_token:
        print("INFO: HF_TOKEN is not set — skipping all model pre-downloads.")
        print("  • Public models (embeddings, reranker) would download fine without a token,")
        print("    but are skipped here to keep CI builds fast (no 2 GB downloads).")
        print("  • Models will be downloaded lazily at runtime on first request.")
        print("  • For production builds, set HF_TOKEN as a Docker build secret.")
        print()
        print("=" * 60)
        print("Skipped (no HF_TOKEN). Image will download models at runtime.")
        print("=" * 60)
        sys.exit(0)

    # ── 1. Classifier (BioBERT — private/pending repo) ──────────────────────────────────
    # This repo may not yet exist on the Hub if the model hasn't been uploaded.
    # The runtime classifier.py already handles this gracefully:
    #   local disk → HF Hub (primary) → DistilBERT fallback
    # A missing Hub repo is therefore NON-FATAL here.
    print("[1/3] Downloading classifier model...")
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    print(f"       Model: {CLASSIFIER_REPO}")
    classifier_ok = False
    try:
        _ = AutoTokenizer.from_pretrained(CLASSIFIER_REPO, token=hf_token)
        _ = AutoModelForSequenceClassification.from_pretrained(
            CLASSIFIER_REPO, token=hf_token
        )
        print(f"       \u2714 Classifier cached ({_format_size(hf_home)})")
        classifier_ok = True
    except Exception as e:
        err_type = type(e).__name__
        print(f"       \u26a0  Classifier not found on Hub ({err_type}).")
        print(f"          Primary repo '{CLASSIFIER_REPO}' returned: {e}")
        print(f"          Trying fallback: {CLASSIFIER_FALLBACK_REPO}")
        try:
            _ = AutoTokenizer.from_pretrained(CLASSIFIER_FALLBACK_REPO, token=hf_token)
            _ = AutoModelForSequenceClassification.from_pretrained(
                CLASSIFIER_FALLBACK_REPO, token=hf_token
            )
            print(f"       \u2714 Fallback classifier cached ({_format_size(hf_home)})")
            classifier_ok = True
        except Exception as e2:
            print(f"       \u26a0  Fallback also unavailable ({type(e2).__name__}): {e2}")
            print()
            print("  \u250c─ ACTION REQUIRED ─────────────────────────────────────────────┐")
            print(f"  \u2502 Push your fine-tuned BioBERT weights to HuggingFace Hub:        \u2502")
            print(f"  \u2502   cd models/classifier/biobert_classifier                       \u2502")
            print(f"  \u2502   huggingface-cli upload {CLASSIFIER_REPO} . --repo-type model  \u2502")
            print("  \u2502                                                                 \u2502")
            print("  \u2502 At runtime the service will fall back to local disk weights     \u2502")
            print("  \u2502 or classify all queries as 'General' without a classifier.     \u2502")
            print("  \u2514───────────────────────────────────────────────────────────────͏┘")
            print()

    # ── 2. Embedding model (PubMedBERT — public repo) ────────────────────────────────
    print()
    print("[2/3] Downloading embedding model...")
    from sentence_transformers import SentenceTransformer

    print(f"       Model: {EMBEDDING_MODEL}")
    try:
        _ = SentenceTransformer(EMBEDDING_MODEL)
        print(f"       \u2714 Embedding model cached ({_format_size(hf_home)})")
    except Exception as e:
        print(f"       \u2717 FATAL: Embedding model download failed: {type(e).__name__}: {e}")
        print("         The embedding model is required — the RAG pipeline cannot run without it.")
        sys.exit(1)

    # ── 3. Reranker (CrossEncoder — public repo) ─────────────────────────────────
    print()
    print("[3/3] Downloading reranker model...")
    from sentence_transformers import CrossEncoder

    print(f"       Model: {RERANKER_MODEL}")
    try:
        _ = CrossEncoder(RERANKER_MODEL)
        print(f"       \u2714 Reranker cached ({_format_size(hf_home)})")
    except Exception as e:
        print(f"       \u2717 FATAL: Reranker download failed: {type(e).__name__}: {e}")
        print("         The reranker is required — set USE_RERANKER=false to skip it.")
        sys.exit(1)

    # ── Summary ─────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    if classifier_ok:
        print("All models pre-downloaded successfully!")
    else:
        print("Embedding + reranker pre-downloaded. Classifier uses runtime fallback.")
        print("Upload the fine-tuned classifier to HuggingFace Hub to bake it in.")
    print(f"Cache location: {hf_home}")
    print(f"Total size:     {_format_size(hf_home)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
