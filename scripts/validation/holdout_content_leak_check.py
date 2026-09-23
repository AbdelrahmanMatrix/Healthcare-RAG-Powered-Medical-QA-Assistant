# -*- coding: utf-8 -*-
"""Content-level holdout exclusion check (final validation).

The eval_holdout.csv carries no chunk ids, so id-level exclusion cannot be
re-verified from the CSV alone. This script instead checks whether any of the
200 sampled evaluation questions (seed 42) appear verbatim in the indexed
corpus (chunk_mapping.pkl, 209,108 rows). Zero matches supports the claim
that the evaluation sample is disjoint from the retrieval index at content
level (P0.4/P0.5 isolation check).
"""

import json
import re
import sys
from pathlib import Path

# Windows/torch quirk: torch must be imported before pandas/pickle-heavy loads
import torch  # noqa: F401  (import-order side effect)

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

MAPPING_PATH = PROJECT_ROOT / "data" / "embeddings" / "faiss_index" / "chunk_mapping.pkl"
HOLDOUT_PATH = PROJECT_ROOT / "data" / "processed" / "eval_holdout.csv"
OUT_JSON = PROJECT_ROOT / "reports" / "holdout_content_leak_check.json"

WS_RE = re.compile(r"\s+")


def norm(s) -> str:
    if not isinstance(s, str):
        return ""
    return WS_RE.sub(" ", s.strip().lower())


print("Loading indexed corpus (chunk_mapping.pkl, ~735 MB) ...")
mapping = pd.read_pickle(MAPPING_PATH)
n_index = len(mapping)
print(f"Indexed rows: {n_index:,} | columns: {list(mapping.columns)}")

print("Loading holdout + sampling 200 (seed 42) ...")
holdout = pd.read_csv(HOLDOUT_PATH)
sample = holdout.sample(n=min(200, len(holdout)), random_state=42)
print(f"Holdout rows: {len(holdout):,} | eval sample: {len(sample)}")

# Normalized question set from the index (fast membership checks)
idx_questions = set(norm(q) for q in mapping["question"].tolist())
idx_questions.discard("")

q_matches, a_matches, ctx_matches = [], [], []
for _, row in sample.iterrows():
    qn = norm(row["question"])
    an = norm(row.get("answer", ""))
    if qn and qn in idx_questions:
        q_matches.append(row["question"][:80])
    # exact-match against normalized answer and context sets (built lazily)
for col, bucket, name in (("answer", a_matches, "answer"), ("context", ctx_matches, "context")):
    if col in mapping.columns:
        idx_set = set(norm(v) for v in mapping[col].tolist())
        idx_set.discard("")
        for _, row in sample.iterrows():
            v = norm(row.get(col, ""))
            if v and v in idx_set:
                bucket.append(row["question"][:80])

result = {
    "method": "exact normalized (lowercased, whitespace-collapsed) string match of each "
              "of the 200 sampled evaluation rows against the full 209,108-row indexed corpus "
              "(question, answer and context columns, checked separately)",
    "n_eval": len(sample),
    "seed": 42,
    "index_rows": int(n_index),
    "question_exact_matches": len(q_matches),
    "answer_exact_matches": len(a_matches),
    "context_exact_matches": len(ctx_matches),
    "verdict": "PASS — no content overlap between eval sample and index"
               if not (q_matches or a_matches or ctx_matches)
               else "FAIL — overlap detected, see matched sample questions",
    "matched_eval_questions_preview": (q_matches + a_matches + ctx_matches)[:10],
}
OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(result, indent=2, ensure_ascii=False)[:1200])
print(f"\nSaved: {OUT_JSON}")
