from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

class MedicalEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=True)

    def get_embedding_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

if __name__ == "__main__":
    embedder = MedicalEmbedder()
    test_text = ["This is a test medical query."]
    embeddings = embedder.generate_embeddings(test_text)
    print(f"Embedding shape: {embeddings.shape}")
