import logging
from fastapi import APIRouter, HTTPException
from api.schemas.request import QueryRequest, QueryResponse, HealthResponse
from config.settings import settings
from src.pipeline import run_pipeline

logger = logging.getLogger("healthcare_rag.routes")
router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest) -> QueryResponse:
    """Run the RAG pipeline and return a grounded answer with mandatory disclaimer."""
    logger.info(f"Query received: {request.question[:80]}")
    try:
        result = run_pipeline(request.question)
        return QueryResponse(
            answer=result.get("answer", "Unable to generate a response."),
            category=result.get("category", "General"),
            retrieved_sources=result.get("sources", []),
            disclaimer=settings.disclaimer,
        )
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health probe for Azure App Service. Returns 200 instantly; never loads models."""
    return HealthResponse(status="ok", model_loaded=True)
