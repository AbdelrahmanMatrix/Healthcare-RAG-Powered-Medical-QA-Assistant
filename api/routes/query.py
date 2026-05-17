import logging
from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool
from api.schemas.request import QueryRequest, QueryResponse, HealthResponse, SourceCitation
from config.settings import settings
from src.pipeline import run_pipeline

logger = logging.getLogger("healthcare_rag.routes")
router = APIRouter()


@router.post("/query", response_model=QueryResponse)
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
    Health probe for Azure App Service.
    Returns model_loaded=True only after the pipeline singleton is initialised.
    Never triggers model loading itself (fast probe).
    """
    from src.pipeline import _rag
    return HealthResponse(status="ok", model_loaded=_rag is not None)
