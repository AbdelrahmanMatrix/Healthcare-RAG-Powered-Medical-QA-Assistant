"""Fix notebook 05 chunking cell: add trailing \n to each source line."""
import json
from pathlib import Path

path = Path("notebooks/05_embeddings_vectorstore.ipynb")
nb = json.loads(path.read_text(encoding="utf-8"))

# Find the chunking cell (id: 4044b8ff425710c3)
for cell in nb["cells"]:
    if cell.get("id") == "4044b8ff425710c3" and cell["cell_type"] == "code":
        src = cell["source"]
        # Add trailing \n to each line that doesn't already have it
        fixed = []
        for i, line in enumerate(src):
            if not line.endswith("\n") and not (i == len(src) - 1 and line.strip() == ""):
                fixed.append(line + "\n")
            else:
                fixed.append(line)
        cell["source"] = fixed
        print(f"Fixed chunking cell: {len(src)} lines -> {len(fixed)} lines")
        break
else:
    print("ERROR: Chunking cell not found!")
    exit(1)

path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("Saved notebook")
