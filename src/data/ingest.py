import os
import pandas as pd
from src.data.loader import MedicalDataLoader
from src.data.preprocessor import MedicalPreprocessor
from src.rag.embeddings import MedicalEmbedder
from src.rag.vectorstore import MedicalVectorStore
from tqdm import tqdm

def ingest_data(limit=None):
    print("🚀 Starting data ingestion...")

    # 1. Load all datasets
    loader = MedicalDataLoader()
    datasets = loader.load_all()

    if limit:
        for name in datasets:
            datasets[name] = datasets[name].head(limit)

    # 2. Preprocess and unify
    preprocessor = MedicalPreprocessor()
    combined_df = preprocessor.unify_datasets(datasets)
    print(f"✅ Unified dataset: {len(combined_df)} records")

    # 3. Generate embeddings
    embedder = MedicalEmbedder()
    # To handle large datasets, we should batch this. MedicalEmbedder.generate_embeddings already uses show_progress_bar
    print("Generating embeddings (this may take a while)...")
    # For 280k, we might want to do it in chunks, but let's try direct for now or suggest it
    embeddings = embedder.generate_embeddings(combined_df['question'].tolist())

    # 4. Create and save Vector Store
    vs = MedicalVectorStore(embedder.get_embedding_dimension())
    vs.add_embeddings(embeddings, combined_df)

    storage_path = "data/embeddings"
    vs.save(storage_path)
    print(f"✅ Vector store saved to {storage_path}")

if __name__ == "__main__":
    # In a real run, we'd remove the limit. For verification, let's use 1000
    ingest_data(limit=1000)
