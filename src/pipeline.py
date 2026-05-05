"""
Top-level pipeline entry point.
- run_query()    → kept for notebook/M2 test compatibility
- run_pipeline() → called by the FastAPI /query endpoint
"""
from src.classification.classifier import predict
from src.rag.pipeline import build_rag_pipeline

_rag = None  # module-level singleton, loaded once on first request


def _get_rag():
    global _rag
    if _rag is None:
        _rag = build_rag_pipeline()
    return _rag


def run_query(query: str) -> dict:
    """Original pipeline — returns full dict including disclaimer baked into 'answer'."""
    category = predict(query)
    return _get_rag().answer_with_routing(query, category=category)


def run_pipeline(question: str) -> dict:
    """
    M3 API entry point. Called by api/routes/query.py.
    Returns answer WITHOUT disclaimer (the API layer injects it separately).
    Returns source IDs as plain strings (the API schema requires List[str]).
    """
    result = run_query(question)
    raw_sources = result.get("retrieved_sources", [])
    source_ids  = [str(s["chunk_id"]) for s in raw_sources] if raw_sources else []
    return {
        "answer":   result.get("answer_raw", result.get("answer", "")),
        "category": result.get("category", "General"),
        "sources":  source_ids,
    }