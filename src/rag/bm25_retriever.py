"""
BM25 keyword retriever as a complement to FAISS semantic search.
For PubMedQA, BM25 often retrieves the exact paper by keyword match.
"""
import pickle
from pathlib import Path
import numpy as np

try:
    from rank_bm25 import BM25Okapi

    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False


class BM25Retriever:
    def __init__(self, mapping_df):
        if not HAS_BM25:
            raise ImportError("Run: pip install rank-bm25")

        self.mapping_df = mapping_df
        # Tokenize the question field for BM25 (questions are the most specific signal)
        corpus = [
            (row["question"] + " " + row["answer"]).lower().split()
            for _, row in mapping_df.iterrows()
        ]
        self.bm25 = BM25Okapi(corpus)
        print(f"✅ BM25 index built over {len(corpus):,} documents")

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            row = self.mapping_df.iloc[idx]
            results.append({
                "chunk_id": int(idx),
                "question": row["question"],
                "context": row["context"],
                "answer": row["answer"],
                "category": row.get("category", "Unknown"),
                "text_chunk": row["text_chunk"],
                "distance": float(-scores[idx]),  # negative so lower = better (consistent with FAISS)
                "bm25_score": float(scores[idx]),
            })
        return results