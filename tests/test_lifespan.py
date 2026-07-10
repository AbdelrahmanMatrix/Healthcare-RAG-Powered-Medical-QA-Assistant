"""
Tests for api/main.py background initialization.

Covers _background_init() which runs as an asyncio task spawned by
the FastAPI lifespan.

IMPORTANT: _background_init() only handles data artifact download.
ML models (embedding, reranker, classifier) are NOT loaded here —
they load lazily on first /query or via /warmup to avoid OOM on
memory-constrained plans (B2 = 3.5 GB RAM).

Testing strategy:
  - Tests call _background_init() directly (it's a regular async function)
  - The lifespan itself only logs a message and spawns the task; we test
    that separately in TestLifespanYields directly against lifespan()
  - All external dependencies (hub, threadpool) are mocked so tests run
    fast and don't require network access or GPU
"""

import logging
import pytest
from unittest.mock import patch, MagicMock
from fastapi import FastAPI


# ── Helpers ──────────────────────────────────────────────────────────────────


async def _run_sync(func, *args, **kwargs):
    """Replacement for starlette.concurrency.run_in_threadpool.

    Calls the sync function directly in the current thread so tests
    don't need an actual thread pool.
    """
    return func(*args, **kwargs)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def mock_threadpool():
    """Replace run_in_threadpool with a simple sync wrapper for all tests.

    Patches at the source module (starlette.concurrency) because the
    import is lazy (inside _background_init), not at the api.main module
    level.
    """
    with patch("starlette.concurrency.run_in_threadpool", side_effect=_run_sync):
        yield


@pytest.fixture
def mock_data_all_present():
    """check_data_exists returns True for all artifacts."""
    with patch("src.data.hub.check_data_exists") as m:
        m.return_value = {
            "/data/raw/file1.csv": True,
            "/data/processed/file2.csv": True,
            "/data/embeddings/index.faiss": True,
        }
        yield m


@pytest.fixture
def mock_data_missing():
    """check_data_exists returns False for some artifacts."""
    with patch("src.data.hub.check_data_exists") as m:
        m.return_value = {
            "/data/raw/file1.csv": True,
            "/data/processed/file2.csv": False,  # missing
            "/data/embeddings/index.faiss": True,
        }
        yield m


@pytest.fixture
def mock_download_ok():
    """download_all_data returns 100% success."""
    with patch("src.data.hub.download_all_data") as m:
        m.return_value = {"downloaded": 1, "skipped": 2, "failed": 0}
        yield m


@pytest.fixture
def mock_download_partial_fail():
    """download_all_data reports some failures."""
    with patch("src.data.hub.download_all_data") as m:
        m.return_value = {"downloaded": 1, "skipped": 0, "failed": 1}
        yield m


# NOTE: No mock_model_warmup_ok or mock_model_warmup_fail fixtures needed.
# ML model loading was removed from _background_init() to avoid OOM on
# memory-constrained plans. Models load lazily on first /query.


# ── Step 1: Data artifact check + download ────────────────────────────────────


class TestBackgroundInitDataCheck:
    """Coverage for Step 1 of _background_init — data artifact availability."""

    @pytest.mark.asyncio
    async def test_all_data_present_skips_download(
        self, caplog, mock_data_all_present,
    ):
        """When all artifacts exist, a skip message is logged and no download occurs.
        ML models are NOT loaded (lazy loading)."""
        from api.main import _background_init

        caplog.set_level(logging.INFO)

        with patch("src.data.hub.download_all_data") as mock_dl:
            await _background_init()

            mock_dl.assert_not_called()

        assert "All data artifacts already present" in caplog.text
        assert "ML models not pre-loaded" in caplog.text

    @pytest.mark.asyncio
    async def test_missing_data_triggers_download(
        self, caplog, mock_data_missing, mock_download_ok,
    ):
        """When artifacts are missing, download_all_data is called and results logged.
        ML models are NOT loaded (lazy loading)."""
        from api.main import _background_init

        caplog.set_level(logging.INFO)

        await _background_init()

        mock_download_ok.assert_called_once()
        assert "Downloading 1 missing data artifact(s)" in caplog.text
        assert "downloaded=1" in caplog.text
        assert "skipped=2" in caplog.text
        assert "ML models not pre-loaded" in caplog.text

    @pytest.mark.asyncio
    async def test_download_failure_logs_error(
        self, caplog, mock_data_missing, mock_download_partial_fail,
    ):
        """Partial download failure logs an error but does not crash."""
        from api.main import _background_init

        caplog.set_level(logging.ERROR)

        await _background_init()

        mock_download_partial_fail.assert_called_once()
        assert "1 artifact(s) failed to download" in caplog.text

    @pytest.mark.asyncio
    async def test_check_data_raises_error_logs_and_continues(
        self, caplog,
    ):
        """When check_data_exists raises (e.g. network timeout), error is logged
        and init continues (no model warmup to fail)."""
        from api.main import _background_init

        with patch("src.data.hub.check_data_exists", side_effect=RuntimeError("Network timeout")):
            caplog.set_level(logging.ERROR)

            await _background_init()

        assert "Data download step failed" in caplog.text
        # No _get_rag mock to check — models are NOT loaded in background init


# ── Lazy loading log message ───────────────────────────────────────────────


class TestBackgroundInitLazyLoading:
    """ML model loading was removed from _background_init() to avoid OOM.
    Models now load lazily on first /query or via /warmup."""

    @pytest.mark.asyncio
    async def test_logs_lazy_loading_message(self, caplog, mock_data_all_present):
        """Background init logs that models will load lazily."""
        from api.main import _background_init

        caplog.set_level(logging.INFO)

        await _background_init()

        assert "ML models not pre-loaded" in caplog.text
        assert "load lazily" in caplog.text

    @pytest.mark.asyncio
    async def test_does_not_call_get_rag(self, mock_data_all_present):
        """_get_rag() should NOT be called by background init."""
        with patch("src.pipeline._get_rag") as mock_get_rag:
            from api.main import _background_init
            await _background_init()
            mock_get_rag.assert_not_called()


# ── Edge cases ────────────────────────────────────────────────────────────────


class TestBackgroundInitEdgeCases:
    """Graceful degradation when data download fails."""

    @pytest.mark.asyncio
    async def test_data_check_fails_init_completes(self):
        """When data check raises an error, _background_init completes without crashing."""
        from api.main import _background_init

        with patch("src.data.hub.check_data_exists", side_effect=RuntimeError("fail")):
            await _background_init()  # should not raise

    @pytest.mark.asyncio
    async def test_logger_message_format(self, caplog):
        """Verify the startup log message is emitted by the lifespan."""
        from api.main import lifespan

        app = FastAPI()
        caplog.set_level(logging.INFO)

        async with lifespan(app):
            pass

        assert "Healthcare RAG API starting up" in caplog.text
