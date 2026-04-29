"""
Top-level pipeline entry point.

Usage:
    from src.pipeline import run_query
    result = run_query("What are the symptoms of diabetes?")
"""

from src.classification.classifier import predict
from src.rag.pipeline import answer, build_rag_pipeline


def run_query(query: str) -> dict:
    """
    Full integrated pipeline:
    query → classify → category-routed retrieve → generate → disclaimer
    """
    # Classify
    category = predict(query)

    # RAG with routing
    from src.rag.pipeline import _pipeline_instance, RAGPipeline

    global _rag
    try:
        _rag
    except NameError:
        _rag = build_rag_pipeline()

    result = _rag.answer_with_routing(query, category=category)

    return result