"""
HuggingFace Hub data download utilities.

Downloads pre-computed data files (raw CSV, cleaned CSV, labelled CSV,
FAISS index, chunk mapping) from the HuggingFace dataset repo so that
users can skip straight to running notebooks without running all
preprocessing steps locally.

Public API:
    check_data_exists()          → dict[path, bool]
    download_file(remote_path)   → bool
    download_all_data()          → dict {downloaded, skipped, failed}
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# HuggingFace dataset repo where preprocessed data is hosted
HF_DATASET_REPO = "AbdoMatrix/healthcare-rag-data"

# Files to check/download: (relative_path_in_repo, local_path)
REQUIRED_FILES = [
    ("data/raw/pubmedqa_raw.csv",
     PROJECT_ROOT / "data" / "raw" / "pubmedqa_raw.csv"),
    ("data/processed/pubmedqa_cleaned.csv",
     PROJECT_ROOT / "data" / "processed" / "pubmedqa_cleaned.csv"),
    ("data/processed/pubmedqa_labelled.csv",
     PROJECT_ROOT / "data" / "processed" / "pubmedqa_labelled.csv"),
    ("data/embeddings/faiss_index/pubmedqa_index_flatl2.faiss",
     PROJECT_ROOT / "data" / "embeddings" / "faiss_index" / "pubmedqa_index_flatl2.faiss"),
    ("data/embeddings/faiss_index/chunk_mapping.pkl",
     PROJECT_ROOT / "data" / "embeddings" / "faiss_index" / "chunk_mapping.pkl"),
]


def _ensure_dir(path: Path) -> None:
    """Create parent directories for a file path if they don't exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


def check_data_exists() -> dict:
    """
    Check which required data files already exist locally.

    Returns:
        Dict mapping local file path (str) to bool (True = exists).
    """
    status = {}
    for _, local_path in REQUIRED_FILES:
        status[str(local_path)] = local_path.exists()
    return status


def download_file(remote_path: str, local_path: Path) -> bool:
    """
    Download a single file from the HuggingFace dataset repo.

    Args:
        remote_path: Path within the HF dataset repo.
        local_path: Local filesystem path to save to.

    Returns:
        True if download succeeded, False otherwise.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("  ⚠️  huggingface-hub not installed. Run: pip install huggingface-hub")
        return False

    hf_token = os.getenv("HF_TOKEN", None)

    try:
        _ensure_dir(local_path)
        hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=remote_path,
            local_dir=PROJECT_ROOT,
            local_dir_use_symlinks=False,
            token=hf_token,
        )
        if local_path.exists():
            size_mb = local_path.stat().st_size / (1024 * 1024)
            print(f"  ✅ Downloaded: {local_path.relative_to(PROJECT_ROOT)} ({size_mb:.1f} MB)")
            return True
        return False
    except Exception as e:
        print(f"  ❌ Failed: {remote_path} — {e}")
        return False


def download_all_data() -> dict:
    """
    Download all missing data files from HuggingFace.

    Returns:
        Dict with keys: downloaded (count), skipped (count), failed (count).
    """
    downloaded = 0
    skipped = 0
    failed = 0

    status = check_data_exists()

    for remote_path, local_path in REQUIRED_FILES:
        if status[str(local_path)]:
            print(f"  ⏭️  Skipped (exists): {local_path.relative_to(PROJECT_ROOT)}")
            skipped += 1
            continue

        success = download_file(remote_path, local_path)
        if success:
            downloaded += 1
        else:
            failed += 1

    return {
        "downloaded": downloaded,
        "skipped": skipped,
        "failed": failed,
    }
