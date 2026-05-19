"""
Run once to upload trained BioBERT weights to HuggingFace Hub.
After this, the Docker container can download them in Azure.

Usage:
    huggingface-cli login   # paste your token
    python scripts/upload_classifier_to_hub.py
"""
from pathlib import Path
from huggingface_hub import HfApi, create_repo

# Updated: BioBERT classifier repo
HF_REPO_ID  = "AbdoMatrix/biobert-medical-classifier"
LOCAL_PATH  = Path("models/classifier/biobert_classifier")

# Fallback: if you trained DistilBERT and haven't retrained with BioBERT yet
DISTILBERT_LOCAL_PATH = Path("models/classifier/distilbert_classifier")
DISTILBERT_HF_REPO    = "AbdoMatrix/distilbert-medical-classifier"


def upload_model(local_path: Path, hf_repo_id: str):
    if not local_path.exists():
        raise FileNotFoundError(
            f"{local_path} not found. Train the model first (notebook 07)."
        )
    has_weights = any(
        (local_path / f).exists()
        for f in ["model.safetensors", "pytorch_model.bin"]
    )
    if not has_weights:
        raise FileNotFoundError(
            f"No .safetensors or .bin file in {local_path}. Train the model first."
        )

    print(f"Local weights found in {local_path}:")
    for f in local_path.iterdir():
        print(f"  {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)")

    create_repo(repo_id=hf_repo_id, repo_type="model", exist_ok=True, private=False)
    print(f"\nRepo ready: https://huggingface.co/{hf_repo_id}")

    HfApi().upload_folder(
        folder_path=str(local_path),
        repo_id=hf_repo_id,
        repo_type="model",
        ignore_patterns=["*.pyc", "__pycache__", "checkpoints/"],
    )
    print(f"\n✅ Done. Open https://huggingface.co/{hf_repo_id}")
    print("Confirm it is PUBLIC before telling Eman to build Docker.")


def main():
    # Try BioBERT first, fallback to DistilBERT
    if LOCAL_PATH.exists() and any(
        (LOCAL_PATH / f).exists() for f in ["model.safetensors", "pytorch_model.bin"]
    ):
        print("🧬 Uploading BioBERT classifier...")
        upload_model(LOCAL_PATH, HF_REPO_ID)
    elif DISTILBERT_LOCAL_PATH.exists() and any(
        (DISTILBERT_LOCAL_PATH / f).exists() for f in ["model.safetensors", "pytorch_model.bin"]
    ):
        print("⚠️  BioBERT weights not found, uploading DistilBERT instead...")
        upload_model(DISTILBERT_LOCAL_PATH, DISTILBERT_HF_REPO)
    else:
        print("❌ No trained model found locally.")
        print("   Run notebook 07_classification_model.ipynb to train the model first.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
