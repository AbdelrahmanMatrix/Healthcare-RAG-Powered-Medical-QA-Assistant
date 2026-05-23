"""
Sync notebooks 05, 06, 09, 10 for Finding 7 changes:
  - Notebook 05: Update chunking cell to use RecursiveCharacterTextSplitter
  - Notebook 06: Fix IndexFlatL2 -> IndexFlatIP in markdown, fix top_k assertion
  - Notebook 09: Fix IndexFlatL2 -> IndexFlatIP in markdown and f-string
  - Notebook 10: Fix holdout assertion 1000 -> 2000
"""

import json
import sys
from pathlib import Path

# Force UTF-8 output for emoji in print
sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def save(nb, path):
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  [OK] Saved: {path.name} ({len(nb['cells'])} cells)")


def update_nb05(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))

        # Find the chunking cell
        if "Build text chunks: question + context + answer" in src and "text_chunk" in src:
            new_src = [
                "# Build text chunks with RecursiveCharacterTextSplitter (chunk_size=700, overlap=150)",
                "try:",
                "    from langchain.text_splitter import RecursiveCharacterTextSplitter",
                "    HAS_LANGCHAIN = True",
                "except ImportError:",
                "    HAS_LANGCHAIN = False",
                '    print("Warning: install langchain-text-splitters for better chunking")',
                "",
                "if HAS_LANGCHAIN:",
                "    splitter = RecursiveCharacterTextSplitter(",
                "        chunk_size=700,",
                "        chunk_overlap=150,",
                '        separators=["\\n\\n", "\\n", ". ", " ", ""],',
                "    )",
                "",
                "    all_chunks = []",
                "    all_meta = []",
                "",
                "    for _, row in df_train.iterrows():",
                "        full_text = (",
                '            f"Question: {row[\'question\']}\\n"',
                '            f"Context: {row[\'context\']}\\n"',
                '            f"Answer: {row[\'answer\']}"',
                "        )",
                "        sub_chunks = splitter.split_text(full_text)",
                "        for chunk in sub_chunks:",
                "            all_chunks.append(chunk)",
                "            all_meta.append({",
                '                "question": row["question"],',
                '                "context": row["context"],',
                '                "answer": row["answer"],',
                '                "category": row.get("category", "Unknown"),',
                '                "text_chunk": chunk,',
                "            })",
                "",
                "    df_chunks = pd.DataFrame(all_meta)",
                "    text_chunks = all_chunks",
                "    n_chunks = len(text_chunks)",
                '    print(f"Built {n_chunks:,} text chunks (from {len(df_train):,} records)")',
                '    print(f"   Sample:\\n{text_chunks[0][:300]}")',
                "else:",
                "    # Fallback: flat concatenation",
                '    df_train["text_chunk"] = (',
                '        "Question: " + df_train["question"] + "\\n"',
                '        + "Context: " + df_train["context"] + "\\n"',
                '        + "Answer: " + df_train["answer"]',
                "    )",
                '    df_chunks = df_train[["question", "context", "answer", "category", "text_chunk"]].copy()',
                '    text_chunks = df_train["text_chunk"].tolist()',
                "    n_chunks = len(text_chunks)",
                '    print(f"Built {n_chunks:,} text chunks")',
                '    print(f"   Sample:\\n{text_chunks[0][:300]}")',
            ]
            cell["source"] = new_src
            print("  [NB05] Updated chunking cell")

        # Update mapping_df -> df_chunks
        if 'mapping_df = df_train[["question", "context", "answer", "category", "text_chunk"]].copy()' in src:
            cell["source"] = [
                line.replace(
                    'mapping_df = df_train[["question", "context", "answer", "category", "text_chunk"]].copy()',
                    'mapping_df = df_chunks.copy()'
                ) for line in cell["source"]
            ]
            print("  [NB05] Updated mapping_df -> df_chunks")

        # Update assertion
        if "index.ntotal == len(mapping_df) == len(df_train)" in src:
            cell["source"] = [
                line.replace("len(df_train)", "len(df_chunks)")
                for line in cell["source"]
            ]
            print("  [NB05] Updated assertion to df_chunks")

        # Update markdown: "L2-normalize" -> "normalize"
        if "L2-normalize" in src:
            cell["source"] = [line.replace("L2-normalize", "normalize") for line in cell["source"]]
            print("  [NB05] 'L2-normalize' -> 'normalize'")

    save(nb, path)


def update_nb06(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))

        # Fix markdown: IndexFlatL2 -> IndexFlatIP
        if "IndexFlatL2" in src:
            cell["source"] = [line.replace("IndexFlatL2", "IndexFlatIP") for line in cell["source"]]
            print("  [NB06] 'IndexFlatL2' -> 'IndexFlatIP'")

        # Fix top_k == 5 assertion
        if "All with 5 sources:" in src and "print" in src:
            cell["source"] = [
                line.replace(
                    "print(f\"All with 5 sources:  {all(r['top_k'] == 5 for r in results)}\")",
                    "print(f\"All with sources:  {all(r['top_k'] >= 5 for r in results)}\")"
                ) for line in cell["source"]
            ]
            print("  [NB06] Fixed top_k assertion")

        # Fix summary message
        if "5 sources retrieved" in src:
            cell["source"] = [
                line.replace("5 sources retrieved", "sources retrieved") for line in cell["source"]
            ]
            print("  [NB06] Fixed summary message")

    save(nb, path)


def update_nb09(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))

        # Fix IndexFlatL2 -> IndexFlatIP
        if "IndexFlatL2" in src:
            cell["source"] = [line.replace("IndexFlatL2", "IndexFlatIP") for line in cell["source"]]
            print("  [NB09] 'IndexFlatL2' -> 'IndexFlatIP'")

        # Fix Chunk format description
        if "Chunk = Q + Context + Answer" in src:
            cell["source"] = [
                line.replace(
                    "Chunk = Q + Context + Answer",
                    "Chunk = RecursiveCharacterTextSplitter (700/150)"
                ) for line in cell["source"]
            ]
            print("  [NB09] Updated chunk format description")

    save(nb, path)


def update_nb10(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))

        # Fix holdout assertion 1000 -> 2000
        if "Expected 1,000 holdout" in src:
            cell["source"] = [
                line.replace(
                    "assert len(df_holdout) == 1000, f\"Expected 1,000 holdout rows, got {len(df_holdout)}\"",
                    "assert len(df_holdout) == 2000, f\"Expected 2,000 holdout rows, got {len(df_holdout)}\""
                ) for line in cell["source"]
            ]
            print("  [NB10] Fixed holdout assertion 1000 -> 2000")

        # Fix print statement
        if "Eval holdout: 1,000" in src:
            cell["source"] = [line.replace("1,000", "2,000") for line in cell["source"]]
            print("  [NB10] Updated holdout print 1,000 -> 2,000")

    save(nb, path)


if __name__ == "__main__":
    notebooks_dir = PROJECT_ROOT / "notebooks"

    print("=" * 60)
    print("Syncing notebooks for Finding 7 changes...")
    print("=" * 60)

    print("\nNotebook 05 -- Embeddings & Vector Store")
    update_nb05(notebooks_dir / "05_embeddings_vectorstore.ipynb")

    print("\nNotebook 06 -- RAG Pipeline")
    update_nb06(notebooks_dir / "06_rag_pipeline.ipynb")

    print("\nNotebook 09 -- Integrated Pipeline")
    update_nb09(notebooks_dir / "09_integrated_pipeline.ipynb")

    print("\nNotebook 10 -- End-to-End Test")
    update_nb10(notebooks_dir / "10_end_to_end_test.ipynb")

    print("\n" + "=" * 60)
    print("All notebooks synced!")
    print("=" * 60)
