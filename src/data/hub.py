"""
HuggingFace Hub data sync for Healthcare RAG project.

Handles uploading and downloading data files, FAISS index,
and chunk mappings to/from HuggingFace Datasets Hub.

Usage:
    from src.data.hub import upload_all_data, download_all_data
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

HF_DATA_REPO = "AbdoMatrix/healthcare-rag-data" 

# Files to sync between local and HuggingFace
DATA_FILES = [
    ("data/raw/pubmedqa_raw.csv", "raw/pubmedqa_raw.csv"),
    ("data/processed/pubmedqa_cleaned.csv", "processed/pubmedqa_cleaned.csv"),
    ("data/processed/pubmedqa_labelled.csv", "processed/pubmedqa_labelled.csv"),
    ("data/embeddings/faiss_index/pubmedqa_index_flatl2.faiss", "embeddings/pubmedqa_index_flatl2.faiss"),
    ("data/embeddings/faiss_index/chunk_mapping.pkl", "embeddings/chunk_mapping.pkl"),
    ("data/embeddings/faiss_index/chunk_mapping.csv", "embeddings/chunk_mapping.csv"),
]


def upload_file(local_relative_path: str, repo_path: str) -> bool:
    """Upload a single file to HuggingFace."""
    local_full = PROJECT_ROOT / local_relative_path

    if not local_full.exists():
        print(f"  ⚠️  Skipped (not found): {local_relative_path}")
        return False

    try:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(local_full),
            path_in_repo=repo_path,
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
        )
        size_mb = local_full.stat().st_size / 1024 ** 2
        print(f"  ✅ Uploaded: {local_relative_path} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"  ❌ Failed: {local_relative_path} — {e}")
        return False


def upload_all_data() -> dict:
    """Upload all data files to HuggingFace."""
    print(f"📤 Uploading data to: huggingface.co/datasets/{HF_DATA_REPO}")
    print("─" * 60)

    api = HfApi()
    api.create_repo(
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        exist_ok=True,
    )

    results = {"uploaded": 0, "skipped": 0, "failed": 0}

    for local_path, repo_path in DATA_FILES:
        success = upload_file(local_path, repo_path)
        if success:
            results["uploaded"] += 1
        else:
            local_full = PROJECT_ROOT / local_path
            if not local_full.exists():
                results["skipped"] += 1
            else:
                results["failed"] += 1

    print("─" * 60)
    print(f"📊 Results: {results['uploaded']} uploaded, "
          f"{results['skipped']} skipped, {results['failed']} failed")

    return results


def download_file(repo_path: str, local_relative_path: str) -> bool:
    """Download a single file from HuggingFace."""
    local_full = PROJECT_ROOT / local_relative_path

    try:
        os.makedirs(local_full.parent, exist_ok=True)

        downloaded_path = hf_hub_download(
            repo_id=HF_DATA_REPO,
            filename=repo_path,
            repo_type="dataset",
        )

        # Copy to project location
        import shutil
        shutil.copy2(downloaded_path, str(local_full))

        size_mb = local_full.stat().st_size / 1024 ** 2
        print(f"  ✅ Downloaded: {local_relative_path} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"  ❌ Failed: {local_relative_path} — {e}")
        return False


def download_all_data() -> dict:
    """Download all data files from HuggingFace."""
    print(f"📥 Downloading data from: huggingface.co/datasets/{HF_DATA_REPO}")
    print("─" * 60)

    results = {"downloaded": 0, "failed": 0}

    for local_path, repo_path in DATA_FILES:
        success = download_file(repo_path, local_path)
        if success:
            results["downloaded"] += 1
        else:
            results["failed"] += 1

    print("─" * 60)
    print(f"📊 Results: {results['downloaded']} downloaded, "
          f"{results['failed']} failed")

    return results


def check_data_exists() -> dict:
    """Check which data files exist locally."""
    status = {}
    for local_path, _ in DATA_FILES:
        full_path = PROJECT_ROOT / local_path
        status[local_path] = full_path.exists()
    return status