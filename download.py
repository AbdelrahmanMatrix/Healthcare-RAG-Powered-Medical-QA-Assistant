"""
Healthcare RAG — One-Command Data Setup

Run after cloning:
    python download_data.py

Downloads all data files, FAISS index, and chunk mappings
from HuggingFace so you can run any notebook immediately.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.data.hub import download_all_data, check_data_exists


def main():
    print("=" * 60)
    print("🏥 Healthcare RAG — Data Setup")
    print("=" * 60)

    status = check_data_exists()
    existing = sum(1 for v in status.values() if v)
    total = len(status)

    if existing == total:
        print(f"\n✅ All {total} data files already exist.")
        print("   No download needed.")
        return

    print(f"\n📂 Found {existing}/{total} files locally.")
    missing = [f for f, exists in status.items() if not exists]
    for f in missing:
        print(f"   ❌ {f}")

    print(f"\n📥 Downloading {len(missing)} missing files...\n")

    results = download_all_data()

    if results['failed'] == 0:
        print("\n🎉 Setup complete! You can now run any notebook.")
    else:
        print(f"\n⚠️  {results['failed']} files failed.")
        print("   Run notebooks 01-05 to generate them manually.")


if __name__ == "__main__":
    main()