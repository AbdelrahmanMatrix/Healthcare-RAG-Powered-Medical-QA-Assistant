"""
Rebuild FAISS index at full scale (~210k vectors).

Improvements over v1:
  - Deduplicates the dataset by question (removes ~78 duplicate rows)
  - Stratified train/holdout split preserving category proportions
  - FAISS built from training set only (true generalization eval)
  - Holdout saved to eval_holdout.csv for downstream evaluation

Run from project root:  python scripts/rebuild_faiss_index.py
"""

import os
import sys
import pickle
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

# ── Anchor to project root ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

print(f"Project root: {PROJECT_ROOT}")
print("Imports successful\n")

# ── Paths ─────────────────────────────────────────────────────────────────
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "pubmedqa_labelled.csv"
HOLDOUT_PATH = PROJECT_ROOT / "data" / "processed" / "eval_holdout.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "embeddings" / "faiss_index"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH = OUTPUT_DIR / "pubmedqa_index_flatl2.faiss"
MAPPING_CSV = OUTPUT_DIR / "chunk_mapping.csv"
MAPPING_PKL = OUTPUT_DIR / "chunk_mapping.pkl"

EVAL_HOLDOUT_SIZE = 2000
RANDOM_SEED = 42

# ── 1. Load Data ──────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Load Data")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"Loaded: {DATA_PATH}")
print(f"   Total rows: {len(df):,}")
print(f"   Columns: {list(df.columns)}")

# Validate required columns
required_columns = {"question", "context", "answer"}
missing = required_columns - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Drop rows with missing text
before = len(df)
df = df.dropna(subset=["question", "context", "answer"]).copy()
df["question"] = df["question"].astype(str).str.strip()
df["context"] = df["context"].astype(str).str.strip()
df["answer"] = df["answer"].astype(str).str.strip()
df = df[(df["question"] != "") & (df["context"] != "") & (df["answer"] != "")]
df = df.reset_index(drop=True)
print(f"   Rows after cleaning: {len(df):,} (dropped {before - len(df)})")

# ── 2. Deduplicate ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Deduplicate by Question")
print("=" * 60)

before_dedup = len(df)
df = df.drop_duplicates(subset=["question"], keep="first").copy().reset_index(drop=True)
dupes_removed = before_dedup - len(df)
print(f"   Rows before dedup: {before_dedup:,}")
print(f"   Duplicates removed: {dupes_removed}")
print(f"   Rows after dedup: {len(df):,}")
print("   Category distribution:")
for cat, count in df["category"].value_counts().items():
    print(f"     {cat}: {count:,} ({count/len(df)*100:.1f}%)")

# ── 3. Stratified Train/Holdout Split ─────────────────────────────────────
print("\n" + "=" * 60)
print(f"STEP 3: Stratified Holdout Split (n={EVAL_HOLDOUT_SIZE})")
print("=" * 60)

df_train, df_eval = train_test_split(
    df,
    test_size=EVAL_HOLDOUT_SIZE,
    stratify=df["category"],
    random_state=RANDOM_SEED,
)
df_train = df_train.reset_index(drop=True)
df_eval = df_eval.reset_index(drop=True)

print(f"   Training set:  {len(df_train):,} rows")
print(f"   Holdout set:   {len(df_eval):,} rows")
print()

print("   Holdout category distribution:")
for cat, count in df_eval["category"].value_counts().items():
    print(f"     {cat}: {count:,} ({count/len(df_eval)*100:.1f}%)")

df_eval.to_csv(HOLDOUT_PATH, index=False)
print(f"\n   Holdout saved: {HOLDOUT_PATH}")

# ── 4. Build Text Chunks (sentence-aware with overlap) ─────────────────────
print("\n" + "=" * 60)
print("STEP 4: Build Text Chunks — RecursiveCharacterTextSplitter")
print("=" * 60)

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    print("   ⚠️  langchain-text-splitters not installed. Falling back to flat chunks.")
    print("        Install: pip install langchain-text-splitters")

if HAS_LANGCHAIN:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []
    all_meta = []

    for _, row in df_train.iterrows():
        full_text = (
            f"Question: {row['question']}\n"
            f"Context: {row['context']}\n"
            f"Answer: {row['answer']}"
        )
        sub_chunks = splitter.split_text(full_text)
        for chunk in sub_chunks:
            all_chunks.append(chunk)
            all_meta.append({
                "question": row["question"],
                "context": row["context"],
                "answer": row["answer"],
                "category": row.get("category", "Unknown"),
                "text_chunk": chunk,
            })

    df_chunks = pd.DataFrame(all_meta)
    text_chunks = all_chunks
    n_chunks = len(text_chunks)
    print(f"   Built {n_chunks:,} text chunks (from {len(df_train):,} records)")
    print(f"   Sample:\n{text_chunks[0][:300]}")
else:
    # Fallback: flat concatenation
    df_train["text_chunk"] = (
        "Question: " + df_train["question"] + "\n"
        + "Context: " + df_train["context"] + "\n"
        + "Answer: " + df_train["answer"]
    )
    df_chunks = df_train[["question", "context", "answer", "category", "text_chunk"]].copy()
    text_chunks = df_train["text_chunk"].tolist()
    n_chunks = len(text_chunks)
    print(f"   Built {n_chunks:,} text chunks")
    print(f"   Sample:\n{text_chunks[0][:300]}")

# ── 5. Load Model & Generate Embeddings ──────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Load Model & Generate Embeddings")
print("=" * 60)

model_name = "pritamdeka/S-PubMedBert-MS-MARCO"
print(f"   Loading model: {model_name} ...")
model = SentenceTransformer(model_name)
emb_dim = model.get_embedding_dimension()
print(f"   Model loaded. Embedding dimension: {emb_dim}")

print(f"\n   Encoding {n_chunks:,} chunks (batch_size=64) ...")
print("   This will take ~1-2 hours on CPU.")
start_time = time.time()

embeddings = model.encode(
    text_chunks,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
)

encoding_time = time.time() - start_time
embeddings = np.asarray(embeddings, dtype=np.float32)

print("\n   Encoding complete!")
print(f"   Shape: {embeddings.shape}")
print(f"   Dtype: {embeddings.dtype}")
print(f"   Time: {encoding_time:.1f}s ({encoding_time/n_chunks*1000:.1f}ms per chunk)")

# ── 6. Build FAISS Index ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Build FAISS Index")
print("=" * 60)

d = embeddings.shape[1]
index = faiss.IndexFlatIP(d)
index.add(embeddings)

print(f"   Dimension: {d}")
print(f"   Total vectors: {index.ntotal:,}")

# ── 7. Save Everything ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Save Index & Chunk Mapping")
print("=" * 60)

# Save FAISS index
faiss.write_index(index, str(INDEX_PATH))

# Save mapping table (from df_chunks which has one row per sub-chunk)
mapping_df = df_chunks.copy()
mapping_df.insert(0, "chunk_id", np.arange(len(mapping_df), dtype=np.int32))

mapping_df.to_csv(MAPPING_CSV, index=False)

with open(MAPPING_PKL, "wb") as f:
    pickle.dump(mapping_df, f)

print(f"   FAISS index:  {INDEX_PATH}")
print(f"     Vectors:    {index.ntotal:,}")
print(f"     File size:  {os.path.getsize(INDEX_PATH) / 1024**2:.1f} MB")
print(f"   Mapping CSV:  {MAPPING_CSV}")
print(f"   Mapping PKL:  {MAPPING_PKL}")

# Final sanity check
assert index.ntotal == len(mapping_df) == len(df_chunks), \
    f"Mismatch: index={index.ntotal}, mapping={len(mapping_df)}, df_chunks={len(df_chunks)}"
print(f"\n   Sanity check PASSED: {index.ntotal:,} vectors == {len(mapping_df):,} mapping rows")

# ── 8. Quick Sanity Check (5 queries) ─────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: Sanity-Check Retrieval (Top-3) + Latency")
print("=" * 60)

test_queries = [
    "What are effective treatments for irritable bowel syndrome symptoms?",
    "Can hypothyroidism increase risk of fatty liver disease?",
    "Is laparoscopic prostatectomy superior to retropubic surgery?",
    "How accurate is diagnosis of acute otitis media in primary care?",
    "Does secondary isoniazid therapy reduce recurrent tuberculosis in HIV patients?",
]

query_embeddings = model.encode(
    test_queries,
    batch_size=16,
    show_progress_bar=False,
    convert_to_numpy=True,
)
query_embeddings = np.asarray(query_embeddings, dtype=np.float32)

k = 3
latencies = []

for qi, query in enumerate(test_queries):
    start = time.perf_counter()
    distances, indices = index.search(query_embeddings[qi:qi+1], k)
    elapsed_ms = (time.perf_counter() - start) * 1000
    latencies.append(elapsed_ms)

    print(f"\n  Query {qi+1}: {query}")
    print(f"  Latency: {elapsed_ms:.2f}ms")
    for rank in range(k):
        idx = int(indices[0, rank])
        dist = float(distances[0, rank])
        chunk = mapping_df.loc[idx, "text_chunk"]
        print(f"    Top {rank+1} | Chunk {idx} | IP={dist:.4f}")
        print(f"      {chunk[:200]}...")

print("\n  LATENCY SUMMARY")
print(f"    Min:    {min(latencies):.2f}ms")
print(f"    Max:    {max(latencies):.2f}ms")
print(f"    Mean:   {np.mean(latencies):.2f}ms")
print(f"    Median: {np.median(latencies):.2f}ms")
print(f"    Index:  {index.ntotal:,} vectors")

if max(latencies) < 500:
    print("\n  KPI MET: All queries < 500ms")
else:
    print("\n  KPI NOT MET: Some queries exceeded 500ms")

print("\n" + "=" * 60)
print("DONE - FAISS index rebuilt at full scale!")
print("=" * 60)
