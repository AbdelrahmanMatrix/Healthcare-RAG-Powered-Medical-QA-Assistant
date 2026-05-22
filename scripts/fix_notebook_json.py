"""Fix corrupted notebook JSON by escaping control chars inside string values."""
import json
import sys
from pathlib import Path

path = Path("notebooks/05_embeddings_vectorstore.ipynb")

content = path.read_text(encoding="utf-8")

# Replace literal newlines/carriage returns inside JSON strings with escaped versions
# Strategy: find unescaped double quotes to track string boundaries
result_chars = []
i = 0
in_string = False

while i < len(content):
    c = content[i]
    
    # Check for escaped character
    if c == "\\" and i + 1 < len(content):
        result_chars.append(c)
        i += 1
        result_chars.append(content[i])
        i += 1
        continue
    
    # Toggle string state at unescaped double quote
    if c == '"':
        in_string = not in_string
        result_chars.append(c)
        i += 1
        continue
    
    # If inside a string, escape control characters
    if in_string and c in "\r\n":
        if c == "\r":
            result_chars.append("\\r")
        else:  # \n
            result_chars.append("\\n")
        i += 1
        continue
    
    result_chars.append(c)
    i += 1

cleaned = "".join(result_chars)

# Validate and modify
try:
    nb = json.loads(cleaned)
    print(f"JSON VALID: {len(nb['cells'])} cells")
    
    # Find and clear the upload cell's outputs
    for ci, cell in enumerate(nb["cells"]):
        src_text = "".join(cell["source"])
        if "upload_file" in src_text and "HuggingFace" in src_text:
            cell["outputs"] = []
            cell["execution_count"] = None
            # Keep metadata but remove ExecuteTime
            if "metadata" in cell and "ExecuteTime" in cell["metadata"]:
                del cell["metadata"]["ExecuteTime"]
            print(f"Cell {ci}: cleared outputs + execution_count")
            print(f"  Has load_dotenv(): {'load_dotenv' in src_text}")
            break
    
    # Write back with proper JSON formatting
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {path}")
    
except json.JSONDecodeError as e:
    print(f"Still invalid JSON at pos {e.pos}: {e}")
    ctx = cleaned[max(0, e.pos-60):e.pos+60]
    print(f"Context: {repr(ctx)}")
    sys.exit(1)
