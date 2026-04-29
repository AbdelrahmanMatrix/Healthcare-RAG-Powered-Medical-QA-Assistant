"""
Embedding utilities for the RAG pipeline.

Wraps SentenceTransformer for consistent usage across the project.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingModel:
    """Wrapper around SentenceTransformer for embedding generation."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Encode texts into float32 numpy array."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query into float32 numpy array."""
        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
        )
        return np.asarray(embedding, dtype=np.float32)