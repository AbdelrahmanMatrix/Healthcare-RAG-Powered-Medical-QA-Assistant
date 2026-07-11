"""Healthcare RAG — FastAPI application entry point."""
import os
import sys
import time
import asyncio
import logging
import mimetypes
from pathlib import Path
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()  # Must run before config.settings reads os.environ  # noqa: E402

from starlette.responses import Response  # noqa: E402

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from fastapi.middleware.gzip import GZipMiddleware  # noqa: E402
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

# Startup state — updated by background init task so /health can report progress
_startup_state: dict = {
    "data_ready": False,
    "pipeline_ready": False,
    "started_at": None,
}


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
    _startup_state["started_at"] = time.monotonic()
    logger.info("Healthcare RAG API starting up...")

    # Spawn background initialization — yield immediately so uvicorn
    # can start serving HTTP requests (health probes, etc.)
    asyncio.create_task(_background_init())

    yield  # application runs — uvicorn processes HTTP requests

    logger.info("Healthcare RAG API shutting down.")


async def _background_init():
    """
    Initialize the application in the background (non-blocking startup).

    Step: Download missing data artifacts (FAISS index, CSVs) from HuggingFace.

    NOTE: ML models (embedding, reranker, classifier) are NOT loaded here.
    They load lazily on the first `/query` request or proactively via `/warmup`.
    This avoids OOM on memory-constrained App Service Plans (B2 = 3.5 GB RAM)
    where loading all three models simultaneously exceeds available memory.

    On Azure App Service, set USE_RERANKER=false to reduce memory pressure
    when models do load (saves ~500 MB).

    /health reports `data_ready` and `pipeline_ready` flags — pipeline_ready
    stays False here since models load lazily.
    """
    from starlette.concurrency import run_in_threadpool

    # ── Step 1: ensure data artifacts are present ───────────────────────────────
    try:
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
        _startup_state["data_ready"] = True
    except Exception as e:  # pragma: no cover
        logger.error(
            f"⚠️  Data download step failed: {e!r} — "
            "pipeline may not work until artifacts are available."
        )

    # ── Step 2: ML models load lazily — skipped here to avoid OOM ──────────
    logger.info(
        "⏳ ML models not pre-loaded — will load lazily on first /query "
        "or via /warmup endpoint."
    )
    _startup_state["pipeline_ready"] = False


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

# ── Middleware stack ───────────────────────────────────────────────────────────
# Order matters: GZip runs last (outermost), so it compresses whatever the
# inner middleware and route handlers produce.
app.add_middleware(
    GZipMiddleware,
    minimum_size=1024,   # only compress responses > 1 KB (skip tiny JSON)
    compresslevel=6,     # balanced speed vs compression (default 9 is too slow)
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


class LoggingStaticFiles(StaticFiles):
    """
    StaticFiles subclass that logs JS file serving for MIME-type diagnostics.

    Logs a warning when a `.js` file is served with an unexpected Content-Type
    (anything other than `application/javascript` or `text/javascript`),
    and an info line for each successful JS serve.
    """

    def file_response(
        self,
        full_path,  # str in this Starlette version
        stat_result: os.stat_result,
        scope: dict,
    ) -> Response:
        response = super().file_response(full_path, stat_result, scope)
        # full_path is a string path in this Starlette version (not a Path object)
        if full_path.endswith(".js"):
            content_type = response.headers.get("content-type", "").lower()
            filename = os.path.basename(full_path)
            logger.debug(
                "📦 Dashboard JS served: /dashboard/%s (%s bytes, Content-Type: %s)",
                filename, stat_result.st_size, content_type,
            )
            # Warn if MIME type is not JavaScript (e.g. wrong server config)
            if content_type and "javascript" not in content_type:
                logger.warning(
                    "⚠️  Dashboard JS served with non-JS Content-Type: "
                    "/dashboard/%s → '%s' (expected 'application/javascript')",
                    filename, content_type,
                )
            # Warn if gzipped (FastAPI GZipMiddleware may double-compress)
            if content_type and "gzip" in content_type:
                logger.warning(
                    "⚠️  Dashboard JS served with gzip Content-Type — "
                    "may indicate double compression: /dashboard/%s",
                    filename,
                )
        return response


# ── Startup static-file diagnostic ────────────────────────────────────
if DASHBOARD_DIR.is_dir():
    js_files = sorted(DASHBOARD_DIR.glob("*.js"))
    if js_files:
        logger.info("📄 Dashboard static files check — %d JS file(s):", len(js_files))
        for f in js_files:
            mime, _ = mimetypes.guess_type(str(f))
            logger.info(
                "   📄 %s (%s bytes, MIME: %s)",
                f.name, f.stat().st_size, mime or "unknown",
            )
    else:
        logger.warning("⚠️  No .js files found in dashboard directory: %s", DASHBOARD_DIR)
else:
    logger.warning("⚠️  Dashboard directory not found: %s", DASHBOARD_DIR)

app.mount("/dashboard", LoggingStaticFiles(directory=str(DASHBOARD_DIR), html=True), name="dashboard")


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
