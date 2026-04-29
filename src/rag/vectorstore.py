"""
FAISS vector store utilities.

Handles index creation, saving, loading, and search.
"""

import pickle
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INDEX_PATH = PROJECT_ROOT / "data" / "embeddings" / "faiss_index" / "pubmedqa_index_flatl2.faiss"
DEFAULT_MAPPING_PATH = PROJECT_ROOT / "data" / "embeddings" / "faiss_index" / "chunk_mapping.pkl"


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """Build a FAISS IndexFlatL2 from embeddings."""
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index


def save_index(index: faiss.IndexFlatL2, path: str) -> None:
    """Save FAISS index to disk."""
    faiss.write_index(index, str(path))


def load_index(path: str = None) -> faiss.IndexFlatL2:
    """Load FAISS index from disk."""
    p = path or str(DEFAULT_INDEX_PATH)
    return faiss.read_index(p)


def load_mapping(path: str = None) -> pd.DataFrame:
    """Load chunk mapping DataFrame from pickle."""
    p = path or str(DEFAULT_MAPPING_PATH)
    with open(p, "rb") as f:
        return pickle.load(f)


def search(index: faiss.IndexFlatL2, query_vector: np.ndarray, k: int = 5):
    """
    Search FAISS index.
    Returns (distances, indices) arrays.
    """
    return index.search(query_vector, k)