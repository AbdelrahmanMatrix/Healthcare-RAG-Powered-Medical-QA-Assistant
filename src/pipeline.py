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


def _expand_query(question: str, category: str) -> str:
    """Expand the query with category context for improved FAISS retrieval.

    Prepends the medical category to the query so the embedding model
    focuses on the relevant medical domain. This helps retrieve chunks
    that are more closely related to the query's medical context.
    """
    if category and category != "General":
        return f"{category.lower()}: {question}"
    return question


def _category_retrieval_quality(retrieved: list) -> float:
    """Compute the average reranker score of the top chunks.

    Used to detect when category-filtered retrieval returns low-quality
    results, triggering a fallback to general (uncategorized) retrieval.

    Args:
        retrieved: List of retrieved chunk dicts with reranker_score.

    Returns:
        Average reranker score of the top-3 chunks, or 0.0 if empty.
    """
    if not retrieved:
        return 0.0
    top_scores = [
        r.get("reranker_score", 0.0)
        for r in retrieved[:3]
        if r.get("reranker_score") is not None
    ]
    if not top_scores:
        return 0.0
    return sum(top_scores) / len(top_scores)


def run_query(query: str) -> dict:
    """
    Notebook / legacy entry point — wraps run_pipeline() so disclaimer
    behaviour is identical to the API path (disclaimer as a separate field,
    NOT baked into the answer string).
    """
    return run_pipeline(query)


def run_pipeline(question: str, top_k: int = None, category: str = None) -> dict:
    """
    M3 API entry point. Called by api/routes/query.py.
    - top_k    : override default retrieval depth (None → use pipeline default)
    - category : force retrieval category (None → classifier infers it)
    Returns answer WITHOUT embedded disclaimer (the API layer injects it).
    Returns source IDs as plain strings + richer source_details list.

    Improvements:
      - Query expansion: category context prepended for better FAISS matching
      - Category-retrieval quality guard: if category-filtered results have
        low reranker scores, falls back to general (uncategorized) retrieval
    """
    rag = _get_rag()

    # predicted_category is saved SEPARATELY from effective_category so the
    # classifier's prediction is always shown in the response, even when the
    # retrieval-quality fallback switches to general (uncategorized) retrieval.
    predicted_category = None

    # Use classifier's per-class probabilities for continuous scoring
    if category:
        effective_category = category
        all_scores = None
    elif rag._use_classifier:
        result = rag._classifier.predict_with_confidence(question)
        predicted_category = result["category"]  # always saved for display
        effective_category = result["category"]
        all_scores = result["all_scores"]
    else:
        predicted_category = predict(question)
        effective_category = predicted_category
        all_scores = None

    # Expand query with category context for improved FAISS retrieval
    expanded_question = _expand_query(question, effective_category)

    if effective_category:
        # Attempt category-prioritised retrieval
        retrieved = rag.retrieve_by_category(
            expanded_question, effective_category, top_k,
            all_scores=all_scores,
        )

        # Fix 2: Fallback when category retrieval has low reranker scores.
        # If the top-3 chunks average reranker score is below the quality
        # threshold (1.0), fall back to general retrieval which may find
        # more relevant chunks even if they don't match the predicted category.
        quality = _category_retrieval_quality(retrieved)
        fallback_threshold = getattr(rag, '_reranker_fallback_threshold', 1.0)
        if retrieved and quality < fallback_threshold:
            retrieved = rag.retrieve(expanded_question, top_k)
            effective_category = None  # category routing wasn't helpful
    else:
        retrieved = rag.retrieve(expanded_question, top_k)

    raw_answer = rag.generate(question, retrieved)
    sources = rag.format_sources(retrieved)
    answer_source = getattr(rag, '_last_answer_source', 'rag')

    # Display the classifier's predicted category even when retrieval fell
    # back to general search (the answer uses better general retrieval, but
    # the user should see the actual predicted topic).
    display_category = predicted_category or effective_category or category or "General"

    return {
        "answer":         raw_answer,
        "category":       display_category,
        "sources":        [str(s["chunk_id"]) for s in sources],
        "source_details": sources,
        "answer_source":  answer_source,
    }
