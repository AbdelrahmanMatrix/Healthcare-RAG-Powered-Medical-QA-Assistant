"""
M3 API tests — run with: pytest tests/test_api.py -v
All 9 tests must pass before Docker build.
"""
import asyncio
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from api.main import app
from config.settings import settings

client = TestClient(app)
MOCK = {"answer": "Mock answer.", "category": "Symptoms", "sources": ["42"]}

VALID_CATEGORIES = {
    "Symptoms", "Diagnosis", "Treatment",
    "Medication", "Prevention", "General",
}


class TestHealth:
    def test_returns_200(self):
        assert client.get("/health").status_code == 200

    def test_status_field_is_ok(self):
        assert client.get("/health").json()["status"] == "ok"

    def test_latency_header_present(self):
        assert "X-Response-Time-Ms" in client.get("/health").headers

    def test_model_loaded_state_before_and_after_query(self):
        """model_loaded=false before any query, true after a mocked query loads the pipeline."""
        mock_rag = MagicMock()
        mock_rag.index.ntotal = 211186

        # Before: pipeline not loaded
        with patch("src.pipeline._rag", None):
            with patch("src.classification.classifier._classifier_instance", None):
                r_before = client.get("/health")
                data_before = r_before.json()
                assert data_before["model_loaded"] is False, \
                    f"Expected model_loaded=False before warmup, got {data_before['model_loaded']}"
                assert data_before["classifier_ready"] is False
                assert data_before["index_vectors"] == 0

        # After: pipeline loaded (simulate a query that loaded _rag)
        with patch("src.pipeline._rag", mock_rag):
            with patch("src.classification.classifier._classifier_instance", MagicMock()):
                r_after = client.get("/health")
                data_after = r_after.json()
                assert data_after["model_loaded"] is True, \
                    f"Expected model_loaded=True after warmup, got {data_after['model_loaded']}"
                assert data_after["classifier_ready"] is True
                assert data_after["index_vectors"] == 211186


class TestQuery:
    @patch("api.routes.query.run_pipeline")
    def test_returns_200(self, m):
        m.return_value = MOCK
        r = client.post("/query", json={"question": "What causes diabetes?"})
        assert r.status_code == 200

    @patch("api.routes.query.run_pipeline")
    def test_all_required_fields_present(self, m):
        m.return_value = MOCK
        data = client.post("/query", json={"question": "How is hypertension treated?"}).json()
        for field in ("answer", "category", "retrieved_sources", "disclaimer"):
            assert field in data, f"Missing field: {field}"

    @patch("api.routes.query.run_pipeline")
    def test_disclaimer_is_non_empty_string(self, m):
        m.return_value = MOCK
        data = client.post("/query", json={"question": "What is metformin used for?"}).json()
        assert isinstance(data["disclaimer"], str) and len(data["disclaimer"]) > 20

    @patch("api.routes.query.run_pipeline")
    def test_category_is_valid(self, m):
        m.return_value = MOCK
        data = client.post("/query", json={"question": "How to prevent stroke?"}).json()
        assert data["category"] in VALID_CATEGORIES

    @patch("api.routes.query.run_pipeline")
    def test_latency_header_present_on_query(self, m):
        m.return_value = MOCK
        r = client.post("/query", json={"question": "What are flu symptoms?"})
        assert "X-Response-Time-Ms" in r.headers

    def test_missing_question_returns_422(self):
        assert client.post("/query", json={}).status_code == 422

    @patch("api.routes.query.run_pipeline")
    def test_pipeline_error_returns_500(self, m):
        m.side_effect = RuntimeError("FAISS index not found")
        r = client.post("/query", json={"question": "What causes anaemia?"})
        assert r.status_code == 500

    @patch("api.routes.query.run_pipeline")
    def test_source_citations_schema(self, m):
        """source_citations must be a list of objects with the SourceCitation fields."""
        m.return_value = MOCK
        data = client.post("/query", json={"question": "What causes anaemia?"}).json()
        assert isinstance(data["source_citations"], list)
        if data["source_citations"]:
            citation = data["source_citations"][0]
            for field in ("chunk_id", "question", "category", "distance"):
                assert field in citation, f"SourceCitation missing field: {field}"

    def test_question_too_short_returns_422(self):
        # min_length=5 on the question field
        assert client.post("/query", json={"question": "hi"}).status_code == 422

    def test_invalid_category_returns_422(self):
        payload = {
            "question": "How is hypertension treated?",
            "category": "Not a category",
        }
        assert client.post("/query", json=payload).status_code == 422

    def test_root_returns_200(self):
        assert client.get("/").status_code == 200


# ==============================================================================
# ── Warmup endpoint ───────────────────────────────────────────────────────────
# ==============================================================================


class TestWarmup:
    """Tests for GET /warmup — pipeline load-on-demand endpoint."""

    def test_warmup_returns_200_with_pipeline_loaded(self):
        """Successful warm-up returns 200 with model_loaded=true."""
        with patch("api.routes.query.run_in_threadpool") as mock_run:
            mock_run.side_effect = lambda fn, *a, **kw: fn(*a, **kw) if callable(fn) else fn
            with patch("src.pipeline._get_rag") as mock_get_rag:
                mock_get_rag.return_value = None
                r = client.get("/warmup")
                assert r.status_code == 200
                data = r.json()
                assert data["status"] == "ok"
                assert "model_loaded" in data
                assert "classifier_ready" in data

    def test_warmup_includes_latency_header(self):
        """Warmup response includes X-Response-Time-Ms header."""
        with patch("api.routes.query.run_in_threadpool") as mock_run:
            mock_run.side_effect = lambda fn, *a, **kw: fn(*a, **kw) if callable(fn) else fn
            with patch("src.pipeline._get_rag") as mock_get_rag:
                mock_get_rag.return_value = None
                r = client.get("/warmup")
                assert "X-Response-Time-Ms" in r.headers


# ==============================================================================
# ── Background Init (api/main.py: _background_init) ───────────────────────────
# ==============================================================================

class TestBackgroundInit:
    """Tests for api/main.py _background_init — data download logic."""

    def test_background_init_data_download_exception(self):
        """_background_init logs error when data download step fails.

        Uses a sync test with asyncio.run() for better coverage tracking.
        """
        from api.main import _background_init

        async def mock_run(func, *args, **kwargs):
            return func(*args, **kwargs)

        with (
            patch("starlette.concurrency.run_in_threadpool", mock_run),
            patch("src.data.hub.check_data_exists") as mock_check,
            patch("src.data.hub.download_all_data") as mock_dl,
            patch("api.main.logger") as mock_logger,
        ):
            mock_check.return_value = {"file.csv": False}
            mock_dl.side_effect = RuntimeError("network error")

            asyncio.run(_background_init())

            error_calls = [str(args[0]) for args, _ in mock_logger.error.call_args_list]
            assert any("Data download step failed" in c for c in error_calls), \
                f"Expected data download error in: {error_calls}"

    def test_background_init_data_already_present(self):
        """_background_init skips download when all data exists."""
        from api.main import _background_init

        async def mock_run(func, *args, **kwargs):
            return func(*args, **kwargs)

        with (
            patch("starlette.concurrency.run_in_threadpool", mock_run),
            patch("src.data.hub.check_data_exists") as mock_check,
            patch("api.main.logger") as mock_logger,
        ):
            mock_check.return_value = {"file.csv": True}

            asyncio.run(_background_init())

            info_calls = [str(args[0]) for args, _ in mock_logger.info.call_args_list]
            assert any("All data artifacts already present" in c for c in info_calls), \
                f"Expected skip message in: {info_calls}"


# ==============================================================================
# ── Auth Middleware ────────────────────────────────────────────────────────────
# ==============================================================================

class TestAuth:
    """Tests for api/middleware/auth.py — API key verification.

    Uses patch.object on the settings singleton so the real verify_api_key
    function is exercised (covers lines 10-13 of auth.py).
    """

    def test_auth_disabled_in_dev(self):
        """When API_KEY is empty/falsy, requests without a key succeed."""
        with patch.object(settings, "API_KEY", ""):
            r = client.post("/query", json={"question": "What causes diabetes?"})
            # Auth passes (no key needed) — may get 500 from pipeline, never 401
            assert r.status_code != 401

    def test_wrong_key_returns_401(self):
        """Request with wrong API key returns 401."""
        with patch.object(settings, "API_KEY", "correct-key-123"):
            r = client.post(
                "/query",
                json={"question": "What causes diabetes?"},
                headers={"X-API-Key": "wrong-key"},
            )
            assert r.status_code == 401

    def test_correct_key_succeeds(self):
        """Request with correct API key passes auth."""
        with patch.object(settings, "API_KEY", "correct-key-123"), \
             patch("api.routes.query.run_pipeline") as mock_run:
            mock_run.return_value = MOCK
            r = client.post(
                "/query",
                json={"question": "What causes diabetes?"},
                headers={"X-API-Key": "correct-key-123"},
            )
            assert r.status_code == 200


# ==============================================================================
# ── Response Caching ──────────────────────────────────────────────────────────
# ==============================================================================

class TestQueryCache:
    """Tests for the in-memory LRU response cache in api/routes/query.py."""

    def setup_method(self):
        """Clear the cache before each test."""
        import api.routes.query as query_mod
        query_mod._cache.clear()

    def test_same_query_twice_hits_cache(self):
        """Querying the same question twice uses cache on the second call."""
        import api.routes.query as query_mod
        query_mod._cache.clear()

        with patch("api.routes.query.run_pipeline") as mock_run:
            mock_run.return_value = MOCK

            # First call — cache miss
            r1 = client.post("/query", json={"question": "What causes diabetes?"})
            assert r1.status_code == 200
            assert mock_run.call_count == 1

            # Second call — cache hit
            r2 = client.post("/query", json={"question": "What causes diabetes?"})
            assert r2.status_code == 200
            # run_pipeline should NOT be called again
            assert mock_run.call_count == 1

    def test_different_queries_no_cache_hit(self):
        """Different questions do NOT share cache entries."""
        import api.routes.query as query_mod
        query_mod._cache.clear()

        with patch("api.routes.query.run_pipeline") as mock_run:
            mock_run.return_value = MOCK

            client.post("/query", json={"question": "What causes diabetes?"})
            client.post("/query", json={"question": "How is hypertension treated?"})
            assert mock_run.call_count == 2

    def test_cache_key_uses_top_k_and_category(self):
        """Cache key includes top_k and category — different params = different cache entry."""
        import api.routes.query as query_mod
        query_mod._cache.clear()

        with patch("api.routes.query.run_pipeline") as mock_run:
            mock_run.return_value = MOCK

            # Same question, different top_k
            client.post("/query", json={"question": "test question abc", "top_k": 5})
            client.post("/query", json={"question": "test question abc", "top_k": 10})
            assert mock_run.call_count == 2

            # Same question, different category
            client.post("/query", json={"question": "test question abc", "category": "Symptoms"})
            client.post("/query", json={"question": "test question abc", "category": "Treatment"})
            assert mock_run.call_count == 4

    def test_cache_entry_expires(self):
        """Cache entries expire after CACHE_TTL seconds."""
        import api.routes.query as query_mod
        query_mod._cache.clear()

        with (
            patch("api.routes.query.run_pipeline") as mock_run,
            patch("api.routes.query.time.monotonic") as mock_time,
        ):
            mock_run.return_value = MOCK
            # Start at time 0
            mock_time.return_value = 0.0

            client.post("/query", json={"question": "What causes diabetes?"})
            assert mock_run.call_count == 1

            # Advance time beyond TTL (301 seconds > 300)
            mock_time.return_value = 301.0

            client.post("/query", json={"question": "What causes diabetes?"})
            # Should be a cache miss — run_pipeline called again
            assert mock_run.call_count == 2

    def test_cache_eviction_lru(self):
        """When cache is full, oldest entry is evicted (LRU)."""
        import api.routes.query as query_mod
        query_mod._cache.clear()

        with patch("api.routes.query.run_pipeline") as mock_run:
            mock_run.return_value = MOCK

            # Set cache size to 2 temporarily
            original_size = query_mod.CACHE_SIZE
            query_mod.CACHE_SIZE = 2
            try:
                # Fill cache with 2 entries
                client.post("/query", json={"question": "test question q1"})
                client.post("/query", json={"question": "test question q2"})
                assert mock_run.call_count == 2

                # q1 should be evicted when q3 is added
                client.post("/query", json={"question": "test question q3"})
                assert mock_run.call_count == 3

                # q1 should be a cache miss (evicted) — run_pipeline called again
                client.post("/query", json={"question": "test question q1"})
                assert mock_run.call_count == 4
            finally:
                query_mod.CACHE_SIZE = original_size


# ==============================================================================
# ── Schema / Request Validation ───────────────────────────────────────────────
# ==============================================================================

class TestSchemaValidation:
    """Tests for api/schemas/request.py — category validator edge cases."""

    def test_valid_category_passes(self):
        """A valid category string passes the validator."""
        for cat in ["Symptoms", "Diagnosis", "Treatment", "Medication", "Prevention", "General"]:
            payload = {"question": "What causes diabetes?", "category": cat}
            r = client.post("/query", json=payload)
            assert r.status_code != 422, f"Valid category '{cat}' should not be rejected"

    def test_case_insensitive_category(self):
        """Category validation is case-insensitive."""
        for cat in ["symptoms", "TREATMENT", "DiAgNoSiS", "general"]:
            payload = {"question": "What causes diabetes?", "category": cat}
            r = client.post("/query", json=payload)
            assert r.status_code != 422, f"Case-insensitive category '{cat}' should not be rejected"

    def test_invalid_category_returns_422(self):
        """An invalid category string returns 422."""
        payload = {"question": "How is hypertension treated?", "category": "Not a category"}
        assert client.post("/query", json=payload).status_code == 422

    def test_whitespace_category_treated_as_none(self):
        """Whitespace-only category is treated as None (no validation error)."""
        payload = {"question": "What causes diabetes?", "category": "   "}
        r = client.post("/query", json=payload)
        assert r.status_code != 422, "Whitespace-only category should be treated as None"

    def test_empty_string_category_treated_as_none(self):
        """Empty string category is treated as None."""
        payload = {"question": "What causes diabetes?", "category": ""}
        r = client.post("/query", json=payload)
        assert r.status_code != 422, "Empty category should be treated as None"

    def test_source_citation_defaults(self):
        """SourceCitation model has correct default values."""
        from api.schemas.request import SourceCitation
        sc = SourceCitation(chunk_id="1", question="q", category="General", distance=0.5)
        assert sc.relevance_score == 0.0
        assert sc.excerpt == ""

    def test_validate_category_directly(self):
        """Directly test the field_validator success path."""
        from api.schemas.request import QueryRequest

        req = QueryRequest(question="What causes diabetes? A valid question here.")
        assert req.category is None

        req = QueryRequest(
            question="What causes diabetes? A long enough question here.",
            category="Treatment",
        )
        assert req.category == "Treatment"

    def test_validate_category_title_case(self):
        """The validator title-cases the category."""
        from api.schemas.request import QueryRequest

        req = QueryRequest(
            question="What causes diabetes? This is a long enough question.",
            category="tReAtMeNt",
        )
        assert req.category == "Treatment"

    def test_validate_category_whitespace_trimmed(self):
        """The validator strips whitespace from category values."""
        from api.schemas.request import QueryRequest

        req = QueryRequest(
            question="What causes diabetes? This is a long enough question.",
            category="   Treatment   ",
        )
        assert req.category == "Treatment"
