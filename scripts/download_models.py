"""
Pre-download all HuggingFace models and data artifacts during Docker build.

This script is called ONCE during docker build (RUN step), so model weights
and data files are baked into the Docker image. This eliminates the 2-3 GB
download on every cold start in Azure App Service.

Call order:
    1. SentenceTransformer embedding model (~1 GB)
    2. CrossEncoder reranker model (~400 MB)
    3. BioBERT classifier from HuggingFace (~500 MB)
    4. FAISS index + chunk mapping + CSVs from HuggingFace dataset (~200 MB)

Usage:
    python scripts/download_models.py
"""

import sys
from pathlib import Path

# Fix stdout encoding for Windows
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, ValueError):
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def download_embedding_model():
    """Pre-download PubMedBERT embedding model (sentence-transformers)."""
    model_name = "pritamdeka/S-PubMedBert-MS-MARCO"
    print(f"\n[{'>'*20}] Downloading embedding model: {model_name} [{'<'*20}]")
    from sentence_transformers import SentenceTransformer
    _ = SentenceTransformer(model_name)
    print(f"  ✅ Embedding model cached: {model_name}")
    return True


def download_reranker():
    """Pre-download CrossEncoder reranker model."""
    model_name = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    print(f"\n[{'>'*20}] Downloading reranker: {model_name} [{'<'*20}]")
    from sentence_transformers import CrossEncoder
    _ = CrossEncoder(model_name)
    print(f"  ✅ Reranker cached: {model_name}")
    return True


def download_classifier():
    """Pre-download BioBERT classifier from HuggingFace."""
    repo_ids = [
        "AbdoMatrix/biobert-medical-classifier",
        "AbdoMatrix/distilbert-medical-classifier",
    ]
    for repo_id in repo_ids:
        print(f"\n[{'>'*20}] Downloading classifier: {repo_id} [{'<'*20}]")
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            _ = AutoTokenizer.from_pretrained(repo_id)
            _ = AutoModelForSequenceClassification.from_pretrained(repo_id)
            print(f"  ✅ Classifier cached: {repo_id}")
            return True
        except Exception as e:
            print(f"  ⚠️  Failed to download {repo_id}: {e}")
            continue
    print("  ❌ All classifier repos failed.")
    return False


def download_data_artifacts():
    """Pre-download FAISS index, chunk mapping, and CSVs from HuggingFace dataset."""
    print(f"\n[{'>'*20}] Downloading data artifacts from HuggingFace dataset [{'<'*20}]")
    from src.data.hub import download_all_data, check_data_exists

    status = check_data_exists()
    missing = [p for p, ok in status.items() if not ok]

    if not missing:
        print("  ✅ All data artifacts already present.")
        return True

    results = download_all_data()
    print(f"  Downloaded={results['downloaded']}, skipped={results['skipped']}, failed={results['failed']}")
    return results['failed'] == 0


def download_fallback_llm():
    """Pre-download fallback LLM (flan-t5-base) for when Groq is unavailable."""
    model_name = "google/flan-t5-base"
    print(f"\n[{'>'*20}] Downloading fallback LLM: {model_name} [{'<'*20}]")
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        _ = AutoTokenizer.from_pretrained(model_name)
        _ = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        print(f"  ✅ Fallback LLM cached: {model_name}")
        return True
    except Exception as e:
        print(f"  ⚠️  Failed to download fallback LLM: {e}")
        return False


def main():
    print("=" * 70)
    print("  Healthcare RAG — Pre-Download Models & Data")
    print("  This eliminates 2-3 GB of downloads on every cold start.")
    print("=" * 70)

    # Track results
    results = {}

    # 1. Embedding model
    results["embedding_model"] = download_embedding_model()

    # 2. Reranker
    results["reranker"] = download_reranker()

    # 3. Classifier
    results["classifier"] = download_classifier()

    # 4. Data artifacts (FAISS index, CSVs, etc.)
    results["data_artifacts"] = download_data_artifacts()

    # 5. Fallback LLM
    results["fallback_llm"] = download_fallback_llm()

    # Summary
    print("\n" + "=" * 70)
    print("  Download Summary")
    print("=" * 70)
    all_ok = True
    for name, ok in results.items():
        status = "✅" if ok else "❌"
        print(f"  {status} {name}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\n  🎉 All models and data pre-downloaded successfully!")
        print("  These are now cached in the HuggingFace cache directory")
        print("  and will be baked into the Docker image.")
    else:
        print("\n  ⚠️  Some downloads failed. The build will continue")
        print("  but cold-start may still require network access.")

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
