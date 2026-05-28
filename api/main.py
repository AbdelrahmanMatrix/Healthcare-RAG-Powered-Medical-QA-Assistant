"""Healthcare RAG — FastAPI application entry point."""
import sys
import time
import asyncio
import logging
from contextlib import asynccontextmanager

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # Must run before config.settings reads os.environ  # noqa: E402

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

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
    On startup:
      Spawn a background task to:
        1. Download missing data artifacts (FAISS index + CSVs) from HuggingFace
        2. Warm up the RAG pipeline by loading models from local cache

      Both steps run asynchronously. The lifespan yields IMMEDIATELY so
      uvicorn can serve HTTP requests (including /health probes from Azure)
      while initialization completes in the background.

      NOTE: ML models are pre-downloaded into the Docker image during the
      build phase (pre_download_models.py), so the warm-up step loads from
      local disk — no network downloads at startup.
    """
    logger.info("Healthcare RAG API starting up...")

    # Spawn background initialization — yield immediately so uvicorn
    # can start serving HTTP requests (health probes, etc.)
    asyncio.create_task(_background_init())

    yield  # application runs here — uvicorn can process HTTP requests


async def _background_init():
    """
    Initialize the application in the background.

    Steps (both run after the lifespan yields, so uvicorn starts immediately):
      1. Download missing data artifacts (FAISS index, CSVs) from HuggingFace
      2. Warm up the RAG pipeline by loading models from the local HF cache

    Step 2 is fast because models are pre-downloaded into the Docker image
    (pre_download_models.py at build time) — no network calls, just disk I/O.

    The /health endpoint returns HTTP 200 right away regardless of whether
    initialization has completed.
    """
    # ── Step 1: ensure data artifacts are present ───────────────────────────────
    try:
        from starlette.concurrency import run_in_threadpool
        from src.data.hub import download_all_data, check_data_exists

        status = await run_in_threadpool(check_data_exists)
        missing = [p for p, ok in status.items() if not ok]
        if missing:
            logger.info(f"Downloading {len(missing)} missing data artifact(s) from HuggingFace...")
            results = await run_in_threadpool(download_all_data)
            logger.info(
                f"Data download complete — downloaded={results['downloaded']}, "
                f"skipped={results['skipped']}, failed={results['failed']}"
            )
            if results["failed"]:
                logger.error(
                    f"⚠️  {results['failed']} artifact(s) failed to download. "
                    "Set HF_TOKEN and verify AbdoMatrix/healthcare-rag-data exists."
                )
        else:
            logger.info("✅ All data artifacts already present — skipping download.")
    except Exception as e:  # pragma: no cover — async handler; exercised by coverage_gaps tests; untraceable
        logger.error(
            f"⚠️  Data download step failed: {e!r} — "
            "pipeline may not work until artifacts are available."
        )

    # ── Step 2: warm up the RAG pipeline from local cache ───────────────────
    try:
        from src.pipeline import _get_rag
        await run_in_threadpool(_get_rag)
        logger.info("✅ RAG pipeline loaded from cache — ready for queries")
    except Exception as e:
        logger.warning(
            f"⚠️  Pipeline warm-up failed: {e!r} — "
            "first query will trigger lazy load. Check FAISS index and model paths."
        )


app = FastAPI(
    title="Healthcare RAG Medical Q&A API",
    version="1.2.0",
    description=(
        "RAG-powered medical Q&A grounded in PubMedQA peer-reviewed research. "
        "Every /query response includes a mandatory medical disclaimer.\n\n"
        f"📊 **[Dashboard]({settings.DASHBOARD_URL})** — system KPIs, model performance, and query interface.\n\n"
        "⚡ **First-query latency:** ML models are pre-loaded in the background at startup. "
        "If the first query arrives before this completes, the pipeline loads lazily — "
        "the query still succeeds but with higher latency (~7–15s instead of ~2–3s). "
        "Call `GET /warmup` proactively to load the pipeline before routing user traffic."
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
    logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)  # pragma: no cover — safety net
    return JSONResponse(status_code=500, content={"error": str(exc)})  # pragma: no cover


# ── Serve Dashboard SPA ───────────────────────────────────────────────
DASHBOARD_DIR = Path(__file__).resolve().parent.parent / "dashboard"
app.mount("/dashboard", StaticFiles(directory=str(DASHBOARD_DIR), html=True), name="dashboard")


app.include_router(query.router)


@app.get("/", include_in_schema=False)
async def root():
    return {
        "project": "Healthcare RAG Medical Q&A Assistant",
        "version": "1.2.0",
        "api_docs": "/docs",
        "health": "/health",
        "dashboard": settings.DASHBOARD_URL,
    }
