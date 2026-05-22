"""Validate all synced notebooks are valid JSON."""
import json
from pathlib import Path

paths = [
    "notebooks/05_embeddings_vectorstore.ipynb",
    "notebooks/06_rag_pipeline.ipynb",
    "notebooks/09_integrated_pipeline.ipynb",
    "notebooks/10_end_to_end_test.ipynb",
]

all_ok = True
for p in paths:
    try:
        nb = json.loads(Path(p).read_text(encoding="utf-8"))
        cells = nb["cells"]
        cell_types = [c["cell_type"] for c in cells]
        print(f"  {p}: VALID ({len(cells)} cells: {cell_types[0]}..{cell_types[-1]})")
    except Exception as e:
        print(f"  {p}: INVALID - {e}")
        all_ok = False

if all_ok:
    print("\nAll notebooks valid!")
else:
    exit(1)
