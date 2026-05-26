"""
Tests for api/main.py background initialization.

Covers _background_init() which runs as an asyncio task spawned by
the FastAPI lifespan:
  1. Data artifact check + download from HuggingFace
  2. RAG pipeline + classifier pre-load

Testing strategy:
  - Tests call _background_init() directly (it's a regular async function)
  - The lifespan itself only logs a message and spawns the task; we test
    that separately in TestLifespanYields directly against lifespan()
  - All external dependencies (hub, pipeline, threadpool) are mocked so
    tests run fast and don't require network access or GPU
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


@pytest.fixture
def mock_model_ok():
    """_get_rag loads without error."""
    with patch("src.pipeline._get_rag") as m:
        m.return_value = MagicMock()
        yield m


@pytest.fixture
def mock_model_fail():
    """_get_rag raises a runtime error."""
    with patch("src.pipeline._get_rag") as m:
        m.side_effect = RuntimeError("GPU out of memory")
        yield m


# ── Step 1: Data artifact check + download ────────────────────────────────────


class TestBackgroundInitDataCheck:
    """Coverage for Step 1 of _background_init — data artifact availability."""

    @pytest.mark.asyncio
    async def test_all_data_present_skips_download(
        self, caplog, mock_data_all_present, mock_model_ok,
    ):
        """When all artifacts exist, a skip message is logged and no download occurs."""
        from api.main import _background_init

        caplog.set_level(logging.INFO)

        with patch("src.data.hub.download_all_data") as mock_dl:
            await _background_init()

            mock_dl.assert_not_called()

        assert "All data artifacts already present" in caplog.text
        assert "Models pre-loaded" in caplog.text

    @pytest.mark.asyncio
    async def test_missing_data_triggers_download(
        self, caplog, mock_data_missing, mock_download_ok, mock_model_ok,
    ):
        """When artifacts are missing, download_all_data is called and results logged."""
        from api.main import _background_init

        caplog.set_level(logging.INFO)

        await _background_init()

        mock_download_ok.assert_called_once()
        assert "Downloading 1 missing data artifact(s)" in caplog.text
        assert "downloaded=1" in caplog.text
        assert "skipped=2" in caplog.text
        assert "Models pre-loaded" in caplog.text

    @pytest.mark.asyncio
    async def test_download_failure_logs_error(
        self, caplog, mock_data_missing, mock_download_partial_fail, mock_model_ok,
    ):
        """Partial download failure logs an error but does not crash."""
        from api.main import _background_init

        caplog.set_level(logging.ERROR)

        await _background_init()

        mock_download_partial_fail.assert_called_once()
        assert "1 artifact(s) failed to download" in caplog.text

    @pytest.mark.asyncio
    async def test_check_data_raises_error_logs_and_continues(
        self, caplog, mock_model_ok,
    ):
        """When check_data_exists raises (e.g. network timeout), error is logged and
        model pre-load still runs."""
        from api.main import _background_init

        with patch("src.data.hub.check_data_exists", side_effect=RuntimeError("Network timeout")):
            caplog.set_level(logging.ERROR)

            await _background_init()

        assert "Data download step failed" in caplog.text
        mock_model_ok.assert_called_once()


# ── Step 2: Model pre-load ───────────────────────────────────────────────────


class TestBackgroundInitModelPreload:
    """Coverage for Step 2 of _background_init — RAG pipeline + classifier pre-load."""

    @pytest.mark.asyncio
    async def test_model_preload_success(
        self, caplog, mock_data_all_present, mock_model_ok,
    ):
        """Successful model pre-load logs confirmation."""
        from api.main import _background_init

        caplog.set_level(logging.INFO)

        await _background_init()

        assert "Models pre-loaded" in caplog.text

    @pytest.mark.asyncio
    async def test_model_preload_failure_logs_error(
        self, caplog, mock_data_all_present, mock_model_fail,
    ):
        """When model pre-load fails, error is logged and init continues."""
        from api.main import _background_init

        caplog.set_level(logging.ERROR)

        await _background_init()

        assert "Model pre-load failed" in caplog.text
        assert "GPU out of memory" in caplog.text

    @pytest.mark.asyncio
    async def test_download_ok_model_fails(
        self, caplog, mock_data_missing, mock_download_ok, mock_model_fail,
    ):
        """Download succeeds but model pre-load fails: Step 1 logs success,
        Step 2 logs error, init does not crash."""
        from api.main import _background_init

        caplog.set_level(logging.ERROR)

        await _background_init()

        mock_download_ok.assert_called_once()
        mock_model_fail.assert_called_once()
        assert "Model pre-load failed" in caplog.text
        assert "GPU out of memory" in caplog.text


# ── Edge cases ────────────────────────────────────────────────────────────────


class TestBackgroundInitEdgeCases:
    """Graceful degradation when everything goes wrong."""

    @pytest.mark.asyncio
    async def test_both_steps_fail_init_completes(self):
        """Even when both data check AND model pre-load raise errors,
        _background_init completes without crashing."""
        from api.main import _background_init

        with patch("src.data.hub.check_data_exists", side_effect=RuntimeError("fail")):
            with patch("src.pipeline._get_rag", side_effect=RuntimeError("fail")):
                await _background_init()  # should not raise

    @pytest.mark.asyncio
    async def test_logger_message_format(self, caplog, mock_data_all_present, mock_model_ok):
        """Verify the startup log message is emitted by the lifespan."""
        from api.main import lifespan

        app = FastAPI()
        caplog.set_level(logging.INFO)

        async with lifespan(app):
            pass

        assert "Healthcare RAG API starting up" in caplog.text
