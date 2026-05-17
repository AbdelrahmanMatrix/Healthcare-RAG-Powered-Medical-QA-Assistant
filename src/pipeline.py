"""
Top-level pipeline entry point — thread-safe singleton loading.
- run_query()    → notebook / M2 test compatibility
- run_pipeline() → called by api/routes/query.py
"""
import threading
from src.classification.classifier import predict
from src.rag.pipeline import build_rag_pipeline

_rag = None
_rag_lock = threading.Lock()


def _get_rag():
    """Double-checked locking: only one thread ever builds the pipeline."""
    global _rag
    if _rag is None:
        with _rag_lock:
            if _rag is None:          # re-check inside the lock
                _rag = build_rag_pipeline()
    return _rag


def run_query(query: str) -> dict:
    """Original pipeline — returns full dict with disclaimer baked into 'answer'."""
    category = predict(query)
    return _get_rag().answer_with_routing(query, category=category)


def run_pipeline(question: str, top_k: int = None, category: str = None) -> dict:
    """
    M3 API entry point. Called by api/routes/query.py.
    - top_k    : override default retrieval depth (None → use pipeline default)
    - category : force retrieval category (None → classifier infers it)
    Returns answer WITHOUT embedded disclaimer (the API layer injects it).
    Returns source IDs as plain strings + richer source_details list.
    """
    rag = _get_rag()
    clf_category = predict(question)
    effective_category = category or clf_category   # explicit > inferred

    if effective_category:
        retrieved = rag.retrieve_by_category(question, effective_category, top_k)
    else:
        retrieved = rag.retrieve(question, top_k)

    raw_answer = rag.generate(question, retrieved)
    sources = rag._format_sources(retrieved)

    return {
        "answer":         raw_answer,
        "category":       effective_category or "General",
        "sources":        [str(s["chunk_id"]) for s in sources],
        "source_details": sources,          # richer data consumed by API schema
    }
