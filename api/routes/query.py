import logging
from fastapi import APIRouter, HTTPException, Depends
from starlette.concurrency import run_in_threadpool
from api.schemas.request import QueryRequest, QueryResponse, HealthResponse, SourceCitation
from config.settings import settings
from src.pipeline import run_pipeline
from api.middleware.auth import verify_api_key

logger = logging.getLogger("healthcare_rag.routes")
router = APIRouter()


@router.post("/query", response_model=QueryResponse, dependencies=[Depends(verify_api_key)])
async def handle_query(request: QueryRequest) -> QueryResponse:
    """
    Run the RAG pipeline.
    ML inference is synchronous — offloaded to threadpool so the event loop
    is never blocked. Supports optional top_k and category overrides.
    """
    logger.info(f"Query received: {request.question[:80]!r} | top_k={request.top_k} | category={request.category}")
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
            )
            for s in raw_details
        ]

        return QueryResponse(
            answer=result.get("answer", "Unable to generate a response."),
            category=result.get("category", "General"),
            retrieved_sources=result.get("sources", []),
            source_citations=source_citations,
            disclaimer=settings.disclaimer,
        )
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
