"""
Download the BioBERT medical classifier weights from HuggingFace Hub.

Run this once after cloning to avoid a 30-90s cold-start download on first inference:
    python scripts/download_classifier.py

Requires HF_TOKEN in .env (or environment) if the repo is private.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_PATH = PROJECT_ROOT / "models" / "classifier" / "biobert_classifier"
HF_REPO = "AbdooMatrix/biobert-medical-classifier"

load_dotenv()

WEIGHT_EXTENSIONS = (".bin", ".safetensors")


def weights_present() -> bool:
    if not LOCAL_PATH.exists():
        return False
    return any(f.suffix in WEIGHT_EXTENSIONS for f in LOCAL_PATH.iterdir())


def main():
    print("=" * 60)
    print("🤖 Healthcare RAG — Classifier Download")
    print("=" * 60)

    if weights_present():
        files = [f.name for f in LOCAL_PATH.iterdir() if f.suffix in WEIGHT_EXTENSIONS]
        print(f"\n✅ Classifier weights already present: {files}")
        print("   No download needed.")
        return

    print(f"\n📥 Downloading from HuggingFace: {HF_REPO}")
    hf_token = os.getenv("HF_TOKEN", "")
    if hf_token:
        print("   HF_TOKEN found — using authenticated download.")
    else:
        print("   ⚠️  HF_TOKEN not set — attempting anonymous download.")
        print("      If the repo is private, set HF_TOKEN in your .env file.")

    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=HF_REPO,
            local_dir=str(LOCAL_PATH),
            token=hf_token or None,
        )
        if weights_present():
            print(f"\n✅ Classifier downloaded to: {LOCAL_PATH}")
        else:
            print("\n⚠️  Download completed but no weight files found.")
            print("    Check that the HuggingFace repo contains .bin or .safetensors files.")
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("   Classifier will be downloaded automatically on first inference.")
        sys.exit(1)


if __name__ == "__main__":
    main()
