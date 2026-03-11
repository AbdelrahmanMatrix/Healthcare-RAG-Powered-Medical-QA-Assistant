import faiss
import numpy as np
import os
import pandas as pd
from typing import List, Tuple

class MedicalVectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = pd.DataFrame()

    def add_embeddings(self, embeddings: np.ndarray, metadata: pd.DataFrame):
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension mismatch. Expected {self.dimension}, got {embeddings.shape[1]}")

        self.index.add(embeddings.astype('float32'))
        self.metadata = pd.concat([self.metadata, metadata], ignore_index=True)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[dict]:
        distances, indices = self.index.search(query_embedding.astype('float32'), k)

        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx != -1:
                res = self.metadata.iloc[idx].to_dict()
                res['distance'] = float(distances[0][i])
                results.append(res)
        return results

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        self.metadata.to_parquet(os.path.join(path, "metadata.parquet"))

    @classmethod
    def load(cls, path: str):
        index = faiss.read_index(os.path.join(path, "index.faiss"))
        metadata = pd.read_parquet(os.path.join(path, "metadata.parquet"))

        instance = cls(dimension=index.d)
        instance.index = index
        instance.metadata = metadata
        return instance

if __name__ == "__main__":
    dim = 384
    vs = MedicalVectorStore(dim)
    dummy_embeddings = np.random.random((10, dim)).astype('float32')
    dummy_meta = pd.DataFrame({'text': [f"Sample {i}" for i in range(10)]})

    vs.add_embeddings(dummy_embeddings, dummy_meta)
    print(f"Index size: {vs.index.ntotal}")

    query = np.random.random((1, dim)).astype('float32')
    results = vs.search(query, k=2)
    print(f"Search results: {results}")
