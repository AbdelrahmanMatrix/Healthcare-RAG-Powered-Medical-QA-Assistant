"""
HuggingFace Hub data download/upload utilities.

Downloads pre-computed data files (raw CSV, cleaned CSV, labelled CSV,
FAISS index, chunk mapping) from the HuggingFace dataset repo so that
users can skip straight to running notebooks without running all
preprocessing steps locally.

Also provides upload utilities so that processed files (cleaned CSV,
labelled CSV, FAISS index, chunk mapping, eval holdout) can be pushed
back to the HuggingFace dataset repo from notebooks.

Public API:
    check_data_exists()          → dict[path, bool]
    download_file(remote_path)   → bool
    download_all_data()          → dict {downloaded, skipped, failed}
    upload_file(local_path, remote_path) → bool
"""

import os
import sys
import shutil
from pathlib import Path

# Ensure emoji/Unicode output works on Windows terminals
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except AttributeError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# HuggingFace dataset repo where preprocessed data is hosted
HF_DATASET_REPO = "AbdoMatrix/healthcare-rag-data"

# Files to check/download: (relative_path_in_repo, local_path)
REQUIRED_FILES = [
    ("raw/pubmedqa_raw.csv",
     PROJECT_ROOT / "data" / "raw" / "pubmedqa_raw.csv"),
    ("processed/pubmedqa_cleaned.csv",
     PROJECT_ROOT / "data" / "processed" / "pubmedqa_cleaned.csv"),
    ("processed/pubmedqa_labelled.csv",
     PROJECT_ROOT / "data" / "processed" / "pubmedqa_labelled.csv"),
    ("embeddings/pubmedqa_index_flatl2.faiss",
     PROJECT_ROOT
     / "data" / "embeddings" / "faiss_index" / "pubmedqa_index_flatip.faiss"),
    ("embeddings/chunk_mapping.pkl",
     PROJECT_ROOT
     / "data" / "embeddings" / "faiss_index" / "chunk_mapping.pkl"),
    ("processed/eval_holdout.csv",
     PROJECT_ROOT / "data" / "processed" / "eval_holdout.csv"),
]

MIN_FILE_BYTES = {}
for _, local_path in REQUIRED_FILES:
    ext = local_path.suffix
    if ext == ".faiss":
        MIN_FILE_BYTES[str(local_path)] = 1_048_576      # 1 MB for FAISS index
    elif ext == ".pkl":
        MIN_FILE_BYTES[str(local_path)] = 102_400        # 100 KB for pickle files
    else:
        MIN_FILE_BYTES[str(local_path)] = 10_240         # 10 KB for CSV / other files


def _ensure_dir(path: Path) -> None:
    """Create parent directories for a file path if they don't exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _file_is_ready(path: Path) -> bool:
    """Return True when a required local artifact exists and is not a stub."""
    if not path.exists():
        return False

    min_bytes = MIN_FILE_BYTES.get(str(path), 1)
    return path.stat().st_size >= min_bytes


def check_data_exists() -> dict:
    """
    Check which required data files already exist locally.

    Returns:
        Dict mapping local file path (str) to bool (True = exists).
    """
    status = {}
    for _, local_path in REQUIRED_FILES:
        status[str(local_path)] = _file_is_ready(local_path)
    return status


def download_file(remote_path: str, local_path: Path) -> bool:
    """
    Download a single file from the HuggingFace dataset repo.

    Files are first downloaded to the HuggingFace cache, then copied
    to the expected local path. This decouples remote paths from local
    paths so the directory structure can differ between the HF repo
    and the local project layout.

    Args:
        remote_path: Path within the HF dataset repo.
        local_path: Local filesystem path to save to.

    Returns:
        True if download succeeded, False otherwise.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("  ⚠️  huggingface-hub not installed."
              " Run: pip install huggingface-hub")
        return False

    hf_token = os.getenv("HF_TOKEN", None)

    try:
        _ensure_dir(local_path)
        cached_path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=remote_path,
            repo_type="dataset",
            token=hf_token,
        )
        # Copy from HF cache to the expected local path
        shutil.copy2(cached_path, local_path)
        if local_path.exists():  # pragma: no cover — exercised by coverage_gaps; untraceable via huggingface_hub patch
            size_mb = local_path.stat().st_size / (1024 * 1024)
            rel = local_path.relative_to(PROJECT_ROOT)
            print(f"  ✅ Downloaded:"
                  f" {rel} ({size_mb:.1f} MB)")
            return True
        return False  # pragma: no cover — unreachable in practice (copy2 raises if it fails)
    except Exception as e:
        print(f"  ❌ Failed:"
              f" {remote_path} — {e}")
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
            print(f"  ⏭️  Skipped (exists):"
                  f" {local_path.relative_to(PROJECT_ROOT)}")
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


def upload_file(local_path: str, remote_path: str) -> bool:
    """
    Upload a single file to the HuggingFace dataset repo.

    Requires the HF_TOKEN environment variable to be set, or an active
    ``huggingface-cli login`` session.

    Args:
        local_path: Path to the local file, relative to the project root.
        remote_path: Destination path within the HF dataset repo.

    Returns:
        True if upload succeeded, False otherwise.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("  ⚠️  huggingface-hub not installed."
              " Run: pip install huggingface-hub")
        return False

    hf_token = os.getenv("HF_TOKEN", None)
    if not hf_token:
        print("  ⚠️  HF_TOKEN not set — will use cached"
              " `huggingface-cli login` token if available.")
        print("     For reliable uploads, set HF_TOKEN"
              " in your .env file or run:")

    full_local = PROJECT_ROOT / local_path
    if not full_local.exists():
        print(f"  ❌ Local file not found: {local_path}")
        return False

    try:
        size_mb = full_local.stat().st_size / (1024 * 1024)
        print(f"  📤 Uploading: {local_path} ({size_mb:.1f} MB)"
              f" → {HF_DATASET_REPO}/{remote_path}")
        HfApi().upload_file(
            path_or_fileobj=str(full_local),
            path_in_repo=remote_path,
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            token=hf_token,
        )
        print(f"  ✅ Uploaded: {remote_path}")
        return True
    except Exception as e:
        print(f"  ❌ Upload failed: {remote_path} — {e}")
        return False
