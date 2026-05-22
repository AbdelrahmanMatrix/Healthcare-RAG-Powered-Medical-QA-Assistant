"""
BM25 keyword retriever as a complement to FAISS semantic search.
For PubMedQA, BM25 often retrieves the exact paper by keyword match.

Medical-aware tokeniser: preserves compound terms, removes punctuation,
filters stopwords, and keeps meaningful medical keywords.
"""

import re

import numpy as np

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

# Medical stopwords to exclude from BM25 (keep medical-specific terms)
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "this", "that", "these", "those", "it",
    "its", "we", "our", "they", "their", "from", "by", "as", "not", "no",
    "than", "then", "so", "if", "when", "where", "how", "what", "which",
    "who", "whom", "about", "after", "all", "also", "any", "because",
    "been", "before", "between", "both", "each", "few", "more", "most",
    "much", "must", "other", "some", "such", "only", "over", "into",
    "through", "during", "before", "after", "above", "below", "up",
    "down", "out", "off", "under", "again", "further", "once",
})

_PUNCT_RE = re.compile(r'[^\w\s-]')        # keep hyphens for compound terms
_MULTI_WS_RE = re.compile(r'\s+')


def _tokenize(text: str) -> list[str]:
    """Medical-aware tokeniser: lowercase, remove punctuation, filter stopwords.

    Preserves hyphenated compound terms (e.g. 'non-insulin-dependent')
    which a naive .split() would destroy.
    """
    if not text:
        return []
    text = _PUNCT_RE.sub(' ', text.lower())
    text = _MULTI_WS_RE.sub(' ', text).strip()
    return [t for t in text.split()
            if t not in _STOPWORDS and len(t) > 1]


class BM25Retriever:
    def __init__(self, mapping_df):
        if not HAS_BM25:
            raise ImportError("Run: pip install rank-bm25")

        self.mapping_df = mapping_df

        # Build corpus: question + answer, tokenised with medical-aware tokeniser
        corpus = [
            _tokenize(str(q) + " " + str(a))
            for q, a in zip(
                mapping_df["question"].fillna(""),
                mapping_df["answer"].fillna(""),
            )
        ]
        self.bm25 = BM25Okapi(corpus)
        print(f"[OK] BM25 index built over {len(corpus):,} documents")

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            row = self.mapping_df.iloc[idx]
            results.append({
                "chunk_id":   int(idx),
                "question":   row["question"],
                "context":    row["context"],
                "answer":     row["answer"],
                "category":   row.get("category", "Unknown"),
                "text_chunk": row["text_chunk"],
                "distance":   float(-scores[idx]),   # negative → lower = better (FAISS-consistent)
                "bm25_score": float(scores[idx]),
            })
        return results
