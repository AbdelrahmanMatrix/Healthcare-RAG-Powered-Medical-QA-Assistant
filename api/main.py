import time
import logging
import sys
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from api.routes import query  # noqa: F401

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("healthcare_rag")

app = FastAPI(
    title="Healthcare RAG Medical Q&A API",
    version="1.0.0",
    description=(
        "RAG-powered medical Q&A grounded in PubMedQA peer-reviewed research. "
        "Every /query response includes a mandatory medical disclaimer."
    ),
)


@app.middleware("http")
async def latency_middleware(request: Request, call_next):
    """Measures request latency and injects X-Response-Time-Ms into every response."""
    start = time.perf_counter()
    response = await call_next(request)
    ms = round((time.perf_counter() - start) * 1000, 2)
    response.headers["X-Response-Time-Ms"] = str(ms)
    logger.info(f"{request.method} {request.url.path} → {response.status_code} | {ms}ms")
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"error": str(exc)})


app.include_router(query.router)


@app.get("/", include_in_schema=False)
async def root():
    return {
        "project": "Healthcare RAG Medical Q&A Assistant",
        "docs": "/docs",
        "health": "/health",
        "version": "1.0.0",
    }


@app.on_event("startup")
async def startup():
    logger.info("Healthcare RAG API started. Swagger UI at /docs")
