"""
Fix all issues in notebooks/05_embeddings_vectorstore.ipynb:
  1. Unminify code cells that are single-list-element strings with embedded \n
  2. Add missing `import os` to the HuggingFace upload cell
  3. Clear error output from failed cells
"""

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

NB_PATH = Path("notebooks/05_embeddings_vectorstore.ipynb")
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

fixes_applied = 0


def unminify_source(cell):
    """Split a single-element source list with embedded \\n into proper multi-line list."""
    src = cell["source"]
    if len(src) != 1:
        return False  # Already properly formatted

    text = src[0]
    if "\n" not in text:
        return False  # No newlines = single line of code, leave as-is

    # Split by newline and add each line as separate list element
    lines = text.split("\n")
    # Reconstruct: each line becomes its own element, with \n appended except possibly the last
    new_src = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            new_src.append(line + "\n")
        else:
            # Last line: add \n only if it's not empty
            if line:
                new_src.append(line)
            # If the last line is empty, don't add anything
    cell["source"] = new_src
    return True


# ── Fix 1: Unminify all code cells with single-element source ──────────
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        if unminify_source(cell):
            print(f"  [Cell {i}] Unminified source ({len(cell['source'])} lines)")
            fixes_applied += 1

# ── Fix 2: Add missing `import os` to HuggingFace upload cell (id: 2032fda9) ──
for i, cell in enumerate(nb["cells"]):
    if cell.get("id") == "2032fda9ee26ec2f":
        src = cell["source"]
        # Find the `import sys` line and add `import os` before it
        new_src = []
        import_os_added = False
        for line in src:
            if "import sys" in line and not import_os_added:
                new_src.append("import os\n")
                import_os_added = True
            new_src.append(line)
        # Also make sure `import os` is before the `from dotenv` line
        # in case `import sys` wasn't found
        if not import_os_added:
            new_src = []
            for line in src:
                if "from dotenv" in line and not import_os_added:
                    new_src.append("import os\n")
                    import_os_added = True
                new_src.append(line)
        cell["source"] = new_src
        print(f"  [Cell {i}] Added 'import os' to HuggingFace upload cell")
        fixes_applied += 1

# ── Fix 3: Clear error outputs from cells ──────────────────────────────
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code" and cell.get("outputs"):
        for output in cell["outputs"]:
            if output.get("output_type") == "error":
                # Found a cell with error output – clear all its outputs
                cell["outputs"] = []
                print(f"  [Cell {i}] Cleared error outputs")
                fixes_applied += 1
                break
        else:
            # Check for execute_result with ename (some errors show differently)
            for output in cell["outputs"]:
                if (output.get("output_type") == "execute_result"
                        and output.get("data", {}).get("text/plain", "")
                        .startswith("NameError")):
                    cell["outputs"] = []
                    print(f"  [Cell {i}] Cleared error execute_result outputs")
                    fixes_applied += 1
                    break

# ── Fix 4: Also fix the markdown summary cell to use proper formatting ──
# The last cell has `{n_chunks:,}` which needs the actual value
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "markdown" and "{n_chunks:,}" in "".join(cell.get("source", [])):
        # Replace placeholder with static text
        new_src = []
        for line in cell["source"]:
            line = line.replace("{n_chunks:,}", "209,108")
            new_src.append(line)
        cell["source"] = new_src
        print(f"  [Cell {i}] Replaced {{n_chunks:,}} placeholder with static count")
        fixes_applied += 1

# ── Save ────────────────────────────────────────────────────────────────
NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"\n✅ {fixes_applied} fixes applied to {NB_PATH}")
