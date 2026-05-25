"""
Clean mislabelled training data using the trained BioBERT classifier.

Strategy ① from the analysis: Use the classifier's high-confidence predictions
to audit the keyword-labelled dataset and fix rows where the classifier is
confident (> 0.95) that the label is wrong.

Usage:
    python scripts/clean_labels.py

Output:
    data/processed/pubmedqa_labelled_cleaned.csv  — cleaned dataset
    Console summary of changes per category
"""

import os
import sys
import time
from pathlib import Path

import pandas as pd

# Windows terminal emoji/Unicode support
os.environ["PYTHONIOENCODING"] = "utf-8"

# ── Anchor to project root ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

# ── Config ───────────────────────────────────────────────────────────────
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "pubmedqa_labelled.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "pubmedqa_labelled_cleaned.csv"

CONFIDENCE_THRESHOLD = 0.95    # Minimum confidence to override a label
BATCH_SIZE = 512               # Rows per batch (tune for CPU/GPU memory)
SAMPLE_LIMIT = 0              # Set > 0 for a dry-run on N rows (0 = full dataset)
DRY_RUN = SAMPLE_LIMIT > 0

# ── Load Data ────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Load Labelled Data")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"  Total rows: {len(df):,}")
print(f"  Columns: {list(df.columns)}")

if SAMPLE_LIMIT > 0:
    # Stratified sample to preserve category proportions
    df = df.groupby("category", group_keys=False).apply(
        lambda x: x.sample(
            max(1, int(len(x) * SAMPLE_LIMIT / len(df))),
            random_state=42,
        )
    ).reset_index(drop=True)
    print(f"  Dry-run mode: limited to {len(df):,} rows")

# Build input text matching training format: "question [SEP] context"
df["text"] = df["question"].astype(str) + " [SEP] " + df["context"].astype(str)

print(f"\n  Category distribution:")
for cat, count in df["category"].value_counts().items():
    print(    f"    {cat:15s}: {count:>7,}")

if DRY_RUN:
    OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / f"pubmedqa_labelled_sample{SAMPLE_LIMIT}.csv"
    print(f"\n  ⚠️  DRY RUN MODE — output will be saved to: {OUTPUT_PATH.name}")
    print(f"      This is NOT the full cleaned dataset.")

# ── Load Classifier ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Load BioBERT Classifier")
print("=" * 60)

from src.classification.classifier import MedicalClassifier

clf = MedicalClassifier()
original_counts = df["category"].value_counts().to_dict()

print(f"\n  Device: {clf.device}")
print(f"  Classes: {list(clf.id2label.values())}")

# ── Batch Classification ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Classify — Batch Predict with Confidence")
print("=" * 60)

texts = df["text"].tolist()
n = len(texts)
all_predictions = []
all_confidences = []

batch_count = (n + BATCH_SIZE - 1) // BATCH_SIZE
start_time = time.time()

for b in range(batch_count):
    batch = texts[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
    batch_results = [clf.predict_with_confidence(t) for t in batch]

    for r in batch_results:
        all_predictions.append(r["category"])
        all_confidences.append(r["confidence"])

    elapsed = time.time() - start_time
    rate = ((b + 1) * BATCH_SIZE) / elapsed if elapsed > 0 else 0
    print(
        f"  Batch {b + 1}/{batch_count} "
        f"({(b + 1) * BATCH_SIZE:,}/{n:,} rows, "
        f"{rate:.0f} rows/sec, "
        f"{elapsed:.1f}s elapsed)"
    )

df["predicted_category"] = all_predictions
df["confidence"] = all_confidences

total_time = time.time() - start_time
print(f"\n  Classification complete: {n:,} rows in {total_time:.1f}s "
      f"({n / total_time:.0f} rows/sec)")

# ── Identify Mislabelled Rows ────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Identify Mislabelled Rows (Confidence ≥ 0.95)")
print("=" * 60)

df["label_differs"] = df["category"] != df["predicted_category"]
df["high_conf_differs"] = df["label_differs"] & (df["confidence"] >= CONFIDENCE_THRESHOLD)

# Per-category breakdown of disagreements
print(f"\n  Confidence threshold: ≥ {CONFIDENCE_THRESHOLD}")
print(f"\n  {'Category':<15} {'Original':>8} {'To Fix':>8} {'% of Cat':>9} "
      f"{'→ Predicted':<15}")
print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*9} {'-'*15}")

total_to_fix = 0
for cat in sorted(original_counts.keys()):
    orig_n = original_counts[cat]
    # Rows currently labelled as `cat` that the classifier thinks are wrong
    mislabelled = df[(df["category"] == cat) & (df["high_conf_differs"])]
    n_fix = len(mislabelled)
    total_to_fix += n_fix

    # What are they being re-labelled to?
    if n_fix > 0:
        top_target = mislabelled["predicted_category"].value_counts().idxmax()
    else:
        top_target = "—"

    pct = 100 * n_fix / orig_n if orig_n > 0 else 0
    print(f"  {cat:<15} {orig_n:>8,} {n_fix:>8,} {pct:>8.2f}%  → {top_target:<15}")

print(f"\n  Total rows to re-label: {total_to_fix:,} / {n:,} "
      f"({100 * total_to_fix / n:.2f}%)")

# ── Apply Corrections ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Apply Corrections")
print("=" * 60)

# Before/after counts
print(f"\n  {'Category':<15} {'Before':>8} {'After':>8} {'Change':>8}")
print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8}")

before_counts = df["category"].value_counts()
# Apply fixes
df.loc[df["high_conf_differs"], "category"] = df.loc[df["high_conf_differs"], "predicted_category"]
after_counts = df["category"].value_counts()

for cat in sorted(original_counts.keys()):
    b = before_counts.get(cat, 0)
    a = after_counts.get(cat, 0)
    delta = a - b
    sign = "+" if delta > 0 else ""
    print(f"  {cat:<15} {b:>8,} {a:>8,} {sign}{delta:>+7,}")

print(f"\n  Total rows modified: {total_to_fix:,}")

# ── Save ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Save Cleaned Dataset")
print("=" * 60)

# Drop helper columns used during cleaning
cols_to_drop = ["text", "predicted_category", "confidence",
                "label_differs", "high_conf_differs"]
df_out = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
df_out.to_csv(OUTPUT_PATH, index=False)
if DRY_RUN:
    print(f"  ⚠️  DRY RUN — sample saved to {OUTPUT_PATH.name} for inspection.")
    print(f"      Delete this file after review; re-run with SAMPLE_LIMIT=0 for production.")

print(f"  Saved: {OUTPUT_PATH}")
print(f"  Rows: {len(df_out):,}")
print(f"  Columns: {list(df_out.columns)}")

# ── Final Summary ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Original labels:  {DATA_PATH.name}")
print(f"  Cleaned labels:   {OUTPUT_PATH.name}")
print(f"  Total rows:       {n:,}")
print(f"  Rows re-labelled: {total_to_fix:,} "
      f"({100 * total_to_fix / n:.2f}%)")
print(f"  Confidence min:   {CONFIDENCE_THRESHOLD}")
print(f"  Classifier:       BioBERT (dmis-lab/biobert-v1.1)")
print(f"  Time:             {total_time:.1f}s")
print()
print("  Next steps:")
print(f"    1. Retrain classifier on {OUTPUT_PATH.name}")
print("    2. Rebuild FAISS index with cleaned labels")
print("    3. Re-run evaluation to measure precision gains")
