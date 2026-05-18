"""Healthcare RAG — FastAPI application entry point."""
import sys
import time
import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()  # Must run before config.settings reads os.environ  # noqa: E402

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from config.settings import settings  # noqa: E402
from api.routes import query  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("healthcare_rag")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Pre-load the RAG pipeline + classifier on startup so the first real
    request isn't penalised by cold-start model loading (10–30s).
    Runs in a threadpool to avoid blocking the event loop during startup.
    """
    logger.info("Healthcare RAG API starting — pre-loading models...")
    try:
        from starlette.concurrency import run_in_threadpool
        from src.pipeline import _get_rag
        await run_in_threadpool(_get_rag)
        logger.info("✅ Models pre-loaded. Swagger UI at /docs")
    except Exception as e:
        logger.error(
            f"⚠️  Model pre-load failed: {e!r} — "
            "first request will trigger load. Check FAISS index and model paths."
        )
    yield  # application runs here
    # (add shutdown cleanup here if needed)


app = FastAPI(
    title="Healthcare RAG Medical Q&A API",
    version="1.2.0",
    description=(
        "RAG-powered medical Q&A grounded in PubMedQA peer-reviewed research. "
        "Every /query response includes a mandatory medical disclaimer."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
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
        "version": "1.2.0",
    }
