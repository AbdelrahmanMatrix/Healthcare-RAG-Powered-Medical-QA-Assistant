import logging
import time
from collections import OrderedDict
from hashlib import sha256

from fastapi import APIRouter, HTTPException, Depends
from deep_translator import GoogleTranslator
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel
from api.schemas.request import QueryRequest, QueryResponse, HealthResponse, SourceCitation
from config.settings import settings
from src.pipeline import run_pipeline
from api.middleware.auth import verify_api_key

logger = logging.getLogger("healthcare_rag.routes")
router = APIRouter()

# ── Translation Request/Response schemas ────────────────────────────────────


class TranslateRequest(BaseModel):
    text: str
    source: str = "auto"
    target: str = "en"


class TranslateResponse(BaseModel):
    translated_text: str


# ── In-memory translation cache ─────────────────────────────────────────────
# Small LRU cache for translations to avoid repeated API calls.
_TRANSLATION_CACHE = OrderedDict()
_TRANSLATION_CACHE_SIZE = 256


def _translate_cached(text: str, source: str, target: str) -> str:
    """Translate text using GoogleTranslator with in-memory caching."""
    key = f"{text}||{source}||{target}"
    if key in _TRANSLATION_CACHE:
        _TRANSLATION_CACHE.move_to_end(key)
        return _TRANSLATION_CACHE[key]

    try:
        translator = GoogleTranslator(source=source, target=target)
        result = translator.translate(text)
        if result:
            # Cache result
            if len(_TRANSLATION_CACHE) >= _TRANSLATION_CACHE_SIZE:
                _TRANSLATION_CACHE.popitem(last=False)
            _TRANSLATION_CACHE[key] = result
            return result
        return text
    except Exception as e:
        logger.warning(f"Translation failed: {e}")
        return text


@router.post("/translate", response_model=TranslateResponse)
async def handle_translate(request: TranslateRequest):
    """
    Translate text between languages using Google Translate (free, no API key).
    Supports auto-detection of source language.
    Used by the dashboard to translate Arabic queries to English and responses back to Arabic.
    """
    try:
        result = await run_in_threadpool(
            _translate_cached,
            request.text,
            request.source,
            request.target,
        )
        return TranslateResponse(translated_text=result)
    except Exception as e:
        logger.error(f"Translation endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Translation error: {str(e)}")


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
    """
    import os
    from src.pipeline import _rag
    from src.classification.classifier import _classifier_instance

    rag_loaded = _rag is not None
    clf_ready = _classifier_instance is not None
    groq_configured = bool(os.getenv("GROQ_API_KEY", ""))
    index_vectors = int(_rag.index.ntotal) if rag_loaded else 0

    return HealthResponse(
        status="ok",
        model_loaded=rag_loaded,
        classifier_ready=clf_ready,
        groq_configured=groq_configured,
        index_vectors=index_vectors,
    )
