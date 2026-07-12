import logging
import time
from collections import OrderedDict
from hashlib import sha256

from fastapi import APIRouter, HTTPException, Depends
from starlette.concurrency import run_in_threadpool
from api.schemas.request import QueryRequest, QueryResponse, HealthResponse, SourceCitation
from config.settings import settings
from src.pipeline import run_pipeline
from api.middleware.auth import verify_api_key

logger = logging.getLogger("healthcare_rag.routes")
router = APIRouter()

# ── Response Cache ───────────────────────────────────────────────────────────
# Simple in-memory LRU cache for duplicate queries.
# Reduces redundant Groq LLM API calls and FAISS retrievals.
# Cache entries expire after CACHE_TTL seconds.

CACHE_SIZE = 64          # max cached queries
CACHE_TTL = 300          # seconds (5 min)

_cache = OrderedDict()   # {query_hash: (timestamp, result_dict)}


def _hash_query(question: str, top_k: int, category: str | None) -> str:
    """Hash the query parameters for cache key."""
    key = f"{question}||{top_k}||{category}"
    return sha256(key.encode()).hexdigest()


def _cache_get(key: str) -> dict | None:
    """Get cached result if fresh."""
    if key not in _cache:
        return None
    ts, result = _cache[key]
    if time.monotonic() - ts > CACHE_TTL:
        del _cache[key]
        return None
    # Move to end (LRU: most recently used)
    _cache.move_to_end(key)
    return result


def _cache_set(key: str, result: dict):
    """Cache a result, evicting oldest if at capacity."""
    if len(_cache) >= CACHE_SIZE:
        _cache.popitem(last=False)  # evict least recently used
    _cache[key] = (time.monotonic(), result)


@router.post("/query", response_model=QueryResponse, dependencies=[Depends(verify_api_key)])
async def handle_query(request: QueryRequest) -> QueryResponse:
    """
    Run the RAG pipeline.
    ML inference is synchronous — offloaded to threadpool so the event loop
    is never blocked. Supports optional top_k and category overrides.

    Responses are cached in-memory (LRU, 64 entries, 5 min TTL) to reduce
    redundant Groq LLM API calls for identical questions.
    """
    logger.info(f"Query received: {request.question[:80]!r} | top_k={request.top_k} | category={request.category}")

    # Check cache first
    cache_key = _hash_query(request.question, request.top_k, request.category)
    cached = _cache_get(cache_key)
    if cached is not None:
        logger.info(f"Cache HIT for query: {request.question[:60]!r}")
        return QueryResponse(
            answer=cached["answer"],
            category=cached["category"],
            retrieved_sources=cached["sources"],
            source_citations=cached["source_citations"],
            disclaimer=cached["disclaimer"],
        )

    logger.info(f"Cache MISS for query: {request.question[:60]!r}")

    try:
        result = await run_in_threadpool(
            run_pipeline,
            request.question,
            request.top_k,
            request.category,
        )

        # Build rich source citations from detail data (backward-compat: also keep str list)
        raw_details = result.get("source_details", [])
        source_citations = [
            SourceCitation(
                chunk_id=str(s["chunk_id"]),
                question=s.get("question", ""),
                category=s.get("category", "Unknown"),
                distance=round(float(s.get("distance", 0.0)), 4),
                relevance_score=round(float(s.get("relevance_score", 0.0)), 4),
                excerpt=s.get("excerpt", ""),
            )
            for s in raw_details
        ]

        response = QueryResponse(
            answer=result.get("answer", "Unable to generate a response."),
            category=result.get("category", "General"),
            retrieved_sources=result.get("sources", []),
            source_citations=source_citations,
            disclaimer=settings.disclaimer,
        )

        # Cache the response
        _cache_set(cache_key, {
            "answer": response.answer,
            "category": response.category,
            "sources": response.retrieved_sources,
            "source_citations": [s.model_dump() for s in response.source_citations],
            "disclaimer": response.disclaimer,
        })

        return response
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health probe — returns diagnostic info about every model component.
    Never triggers model loading (fast, safe to call frequently).

    **Cold-start / lazy loading fallback:**
    - `model_loaded=false` until the background warm-up completes or the
      first query triggers lazy loading. This is normal at startup.
    - `classifier_ready=false` until the BioBERT classifier is loaded.
    - The first `/query` call will trigger lazy loading if warm-up hasn't
      finished — it succeeds but with higher latency (~7-15s vs ~2-3s).
    - Call `GET /warmup` to proactively trigger loading.
    """
    import os
    from src.pipeline import _rag
    from src.classification.classifier import _classifier_instance

    rag_loaded = _rag is not None
    clf_ready = _classifier_instance is not None
    groq_configured = bool(os.getenv("GROQ_API_KEY", ""))
    index_vectors = int(_rag.index.ntotal) if rag_loaded else 0

    from api.main import _startup_state
    data_ready = _startup_state.get("data_ready", False)
    # "initializing" during cold-start, "ok" when both data and pipeline are ready
    if data_ready and rag_loaded:
        status = "ok"  # pragma: no cover — covered by integration health-check tests
    else:
        status = "initializing"

    return HealthResponse(
        status=status,
        model_loaded=rag_loaded,
        classifier_ready=clf_ready,
        groq_configured=groq_configured,
        index_vectors=index_vectors,
    )


@router.get("/warmup")
async def warmup():
    """
    Warm up the RAG pipeline by loading it into memory.

    Models are pre-cached in the Docker image (pre_download_models.py), so
    this loads from local disk — no network downloads. Blocks until the
    pipeline is ready, then returns a 200 response.

    Useful for:
      - Azure deployment scripts that call /warmup before routing traffic
      - Proactive startup warm-up to ensure the first user query is fast
      - Monitoring probes that also want to trigger model loading

    **Lazy-loading fallback:**
    If the background warm-up hasn't completed yet, calling this endpoint
    triggers the lazy load explicitly — subsequent queries will have normal
    warm latency (~2-3s).

    This endpoint is NOT required for normal operation — the pipeline loads
    lazily on the first /query call and the lifespan pre-loads it in the
    background automatically.
    """
    from src.pipeline import _get_rag
    await run_in_threadpool(_get_rag)

    from src.pipeline import _rag
    from src.classification.classifier import _classifier_instance

    return {
        "status": "ok",
        "model_loaded": _rag is not None,
        "classifier_ready": _classifier_instance is not None,
    }
