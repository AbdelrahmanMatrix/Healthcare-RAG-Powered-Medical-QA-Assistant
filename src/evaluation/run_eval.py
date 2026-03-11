import os
import pandas as pd
from src.data.loader import MedicalDataLoader
from src.data.preprocessor import MedicalPreprocessor
from src.rag.embeddings import MedicalEmbedder
from src.rag.vectorstore import MedicalVectorStore
from src.rag.pipeline import MedicalRAGPipeline
from src.evaluation.metrics import MedicalEvaluator
import mlflow

def run_evaluation():
    # 1. Load data
    loader = MedicalDataLoader()
    # Using small sample for quick evaluation
    print("Loading data...")
    pubmed_df = loader.load_pubmedqa()
    sample_df = pubmed_df.head(10)

    # 2. Preprocess
    print("Preprocessing...")
    preprocessor = MedicalPreprocessor()
    processed_df = preprocessor.preprocess_pubmedqa(sample_df)

    # 3. Embed & Index
    print("Embedding & Indexing...")
    embedder = MedicalEmbedder()
    embeddings = embedder.generate_embeddings(processed_df['question'].tolist())

    vs = MedicalVectorStore(embedder.get_embedding_dimension())
    vs.add_embeddings(embeddings, processed_df)

    # 4. Pipeline
    pipeline = MedicalRAGPipeline(vs, embedder)

    # 5. Evaluate
    evaluator = MedicalEvaluator()
    references = processed_df['answer'].tolist()
    candidates = []

    print("Running pipeline for each query...")
    for q in processed_df['question']:
        res = pipeline.answer(q)
        # Strip disclaimer for evaluation
        clean_res = res['response'].split("\n\nDISCLAIMER")[0].replace("Based on medical literature: ", "")
        candidates.append(clean_res)

    print("Calculating metrics...")
    metrics = evaluator.evaluate_batch(references, candidates)
    print(f"Evaluation Results: {metrics}")

    # 6. Log to MLflow
    mlflow.set_experiment("Medical_RAG_Evaluation")
    with mlflow.start_run():
        mlflow.log_params({"num_samples": 10, "model": "all-MiniLM-L6-v2"})
        mlflow.log_metrics(metrics)
        print("Logged to MLflow.")

if __name__ == "__main__":
    run_evaluation()
