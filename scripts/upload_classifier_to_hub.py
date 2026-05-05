"""
Run once to upload trained DistilBERT weights to HuggingFace Hub.
After this, the Docker container can download them in Azure.
"""
from pathlib import Path
from huggingface_hub import HfApi, create_repo

HF_REPO_ID = "AbdoMatrix/distilbert-medical-classifier"
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