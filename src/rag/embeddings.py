"""
Embedding utilities for the RAG pipeline.
Domain-specific biomedical embedding model (upgraded from all-MiniLM-L6-v2); now:
pritamdeka/S-PubMedBert-MS-MARCO (biomedical domain).
"""
import numpy as np
from sentence_transformers import SentenceTransformer

# Medical-domain embedding model — trained on PubMed + MS-MARCO retrieval.
DEFAULT_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"


class EmbeddingModel:
    """Wrapper around SentenceTransformer for embedding generation."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_embedding_dimension()
        print(f"✅ Embedding model loaded: {model_name} (dim={self.dimension})")

    def encode(self, texts: list, batch_size: int = 32) -> np.ndarray:
        """Encode texts into float32 numpy array (L2-normalised)."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query into float32 numpy array."""
        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(embedding, dtype=np.float32)
