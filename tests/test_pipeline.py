"""
Unit tests for src/pipeline.py.

Covers:
  - run_query()     → legacy wrapper
  - run_pipeline()  → main API entry point (with/without category, top_k)
  - _get_rag()      → thread-safe singleton loading
"""

from unittest.mock import patch, MagicMock
import pytest


# ==============================================================================
# ── Global-state cleanup ──────────────────────────────────────────────────────
# ── _get_rag() caches its result in a module-level _rag global.  We must
#    reset it before every test so that the patched build_rag_pipeline is
#    actually called (rather than returning the cached value from a prior test).
# ==============================================================================

@pytest.fixture(autouse=True)
def reset_rag_global():
    """Reset _rag singleton before each test so mock pipelines take effect."""
    import src.pipeline
    src.pipeline._rag = None
    yield


# ==============================================================================
# ── Fixtures ──────────────────────────────────────────────────────────────────
# ==============================================================================

@pytest.fixture
def mock_rag():
    """Build a mock RAG object with all methods used by run_pipeline."""
    rag = MagicMock()
    # Make the pipeline fall back to the module-level predict() so tests
    # can continue to mock src.pipeline.predict for category control.
    rag._use_classifier = False
    # Allow category-retrieval quality check (set a valid float threshold)
    rag._reranker_fallback_threshold = 1.0

    def _format_sources(retrieved):
        return [
            {
                "chunk_id": idx,
                "question": r.get("question", f"q{idx}"),
                "category": r.get("category", "General"),
                "distance": r.get("distance", 0.0),
                "relevance_score": r.get("relevance_score", 0.9),
                "excerpt": r.get("excerpt", f"excerpt {idx}"),
            }
            for idx, r in enumerate(retrieved)
        ]

    rag.format_sources = _format_sources
    rag.retrieve.return_value = [
        {"chunk_id": 1, "question": "q1", "category": "General",
         "distance": 0.1, "relevance_score": 0.9, "excerpt": "e1",
         "reranker_score": 2.0},
        {"chunk_id": 2, "question": "q2", "category": "General",
         "distance": 0.2, "relevance_score": 0.8, "excerpt": "e2",
         "reranker_score": 1.5},
    ]
    rag.retrieve_by_category.return_value = [
        {"chunk_id": 10, "question": "q10", "category": "Symptoms",
         "distance": 0.05, "relevance_score": 0.95, "excerpt": "e10",
         "reranker_score": 2.5},
    ]
    rag.generate.return_value = "The patient may experience fever and fatigue."
    return rag


@pytest.fixture
def mock_rag_cls(mock_rag):
    """Patch build_rag_pipeline to return the mock_rag."""
    with patch("src.pipeline.build_rag_pipeline", return_value=mock_rag) as m:
        yield m


# ==============================================================================
# ── _get_rag ──────────────────────────────────────────────────────────────────
# ==============================================================================

class TestGetRag:
    """Tests for _get_rag() singleton behaviour."""

    def test_singleton_returns_same_instance(self, mock_rag_cls, mock_rag):
        """_get_rag() returns the same instance on repeated calls."""
        from src.pipeline import _get_rag

        instance1 = _get_rag()
        instance2 = _get_rag()
        assert instance1 is instance2
        mock_rag_cls.assert_called_once()

    def test_singleton_builds_once(self, mock_rag_cls, mock_rag):
        """build_rag_pipeline is only called once."""
        from src.pipeline import _get_rag

        _get_rag()
        _get_rag()
        _get_rag()
        mock_rag_cls.assert_called_once()


# ==============================================================================
# ── run_query ─────────────────────────────────────────────────────────────────
# ==============================================================================

class TestRunQuery:
    """Tests for run_query() — legacy wrapper."""

    def test_run_query_delegates_to_run_pipeline(self, mock_rag_cls, mock_rag):
        """run_query() returns the same format as run_pipeline()."""
        from src.pipeline import run_query

        with patch("src.pipeline.predict", return_value="General"):
            result = run_query("What are the symptoms of flu?")
        assert isinstance(result, dict)
        assert "answer" in result
        assert "category" in result
        assert "sources" in result
        assert "source_details" in result
        assert result["answer"] == "The patient may experience fever and fatigue."

    def test_run_query_calls_classifier(self, mock_rag_cls, mock_rag):
        """run_query classifies the question when no category is provided."""
        from src.pipeline import run_query

        with patch("src.pipeline.predict", return_value="Diagnosis") as mock_predict:
            result = run_query("How is diabetes diagnosed?")
            mock_predict.assert_called_once_with("How is diabetes diagnosed?")
            assert result["category"] == "Diagnosis"

    def test_run_query_empty_answer(self, mock_rag_cls, mock_rag):
        """run_query handles empty answer from the pipeline."""
        mock_rag.generate.return_value = ""
        from src.pipeline import run_query

        with patch("src.pipeline.predict", return_value="General"):
            result = run_query("test")
        assert result["answer"] == ""


# ==============================================================================
# ── run_pipeline — Basics ─────────────────────────────────────────────────────
# ==============================================================================

class TestRunPipelineBasics:
    """Tests for run_pipeline() structure."""

    def test_returns_expected_keys(self, mock_rag_cls, mock_rag):
        """Return dict has all expected keys."""
        from src.pipeline import run_pipeline

        with patch("src.pipeline.predict", return_value="General"):
            result = run_pipeline("test question")
        assert set(result.keys()) == {"answer", "category", "sources", "source_details", "answer_source"}

    def test_answer_is_string(self, mock_rag_cls, mock_rag):
        """Answer is a string."""
        from src.pipeline import run_pipeline

        with patch("src.pipeline.predict", return_value="General"):
            result = run_pipeline("test")
        assert isinstance(result["answer"], str)

    def test_sources_are_string_ids(self, mock_rag_cls, mock_rag):
        """Sources list contains string chunk IDs."""
        from src.pipeline import run_pipeline

        with patch("src.pipeline.predict", return_value="General"):
            result = run_pipeline("test")
        for s in result["sources"]:
            assert isinstance(s, str)

    def test_source_details_is_list_of_dicts(self, mock_rag_cls, mock_rag):
        """Source details is a list with expected keys."""
        from src.pipeline import run_pipeline

        with patch("src.pipeline.predict", return_value="General"):
            result = run_pipeline("test")
        assert isinstance(result["source_details"], list)
        if result["source_details"]:
            d = result["source_details"][0]
            assert "chunk_id" in d
            assert "question" in d
            assert "category" in d
            assert "distance" in d
            assert "relevance_score" in d
            assert "excerpt" in d

    def test_retrieve_general_when_no_category(self, mock_rag_cls, mock_rag):
        """When classifier returns None, uses general retrieve()."""
        from src.pipeline import run_pipeline

        with patch("src.pipeline.predict", return_value=None):
            result = run_pipeline("test")
            mock_rag.retrieve.assert_called_once()
            mock_rag.retrieve_by_category.assert_not_called()
            assert result["category"] == "General"


# ==============================================================================
# ── run_pipeline — Category handling ──────────────────────────────────────────
# ==============================================================================

class TestRunPipelineCategory:
    """Tests for category inference & override."""

    def test_classifier_infers_category(self, mock_rag_cls, mock_rag):
        """Without explicit category, classifier predicts it."""
        from src.pipeline import run_pipeline

        with patch("src.pipeline.predict", return_value="Treatment") as mock_predict:
            result = run_pipeline("What is the treatment for hypertension?")
            mock_predict.assert_called_once_with(
                "What is the treatment for hypertension?"
            )
            assert result["category"] == "Treatment"
            mock_rag.retrieve_by_category.assert_called_once()

    def test_explicit_category_skips_classifier(self, mock_rag_cls, mock_rag):
        """When category is provided, classifier is NOT called."""
        from src.pipeline import run_pipeline

        with patch("src.pipeline.predict") as mock_predict:
            result = run_pipeline("test", category="Symptoms")
            mock_predict.assert_not_called()
            assert result["category"] == "Symptoms"
            mock_rag.retrieve_by_category.assert_called_once()

    def test_category_empty_string(self, mock_rag_cls, mock_rag):
        """Empty string category falls back to classifier."""
        from src.pipeline import run_pipeline

        with patch("src.pipeline.predict", return_value="Medication") as mock_predict:
            result = run_pipeline("test", category="")
            mock_predict.assert_called_once()
            assert result["category"] == "Medication"

    def test_invalid_category_falls_back_to_general(self, mock_rag_cls, mock_rag):
        """When classifier returns None and no category provided, uses general retrieval."""
        from src.pipeline import run_pipeline

        with patch("src.pipeline.predict", return_value=None):
            result = run_pipeline("test")
            mock_rag.retrieve.assert_called_once()
            mock_rag.retrieve_by_category.assert_not_called()
            assert result["category"] == "General"


# ==============================================================================
# ── run_pipeline — top_k override ─────────────────────────────────────────────
# ==============================================================================

class TestRunPipelineTopK:
    """Tests for top_k parameter passthrough."""

    def test_top_k_passed_to_retrieve_by_category(self, mock_rag_cls, mock_rag):
        """top_k is forwarded to retrieve_by_category."""
        from src.pipeline import run_pipeline

        with patch("src.pipeline.predict", return_value="Diagnosis"):
            run_pipeline("test", top_k=10, category="Diagnosis")
            mock_rag.retrieve_by_category.assert_called_once_with(
                "diagnosis: test", "Diagnosis", 10, all_scores=None
            )

    def test_top_k_passed_to_general_retrieve(self, mock_rag_cls, mock_rag):
        """top_k is forwarded to retrieve when no category."""
        from src.pipeline import run_pipeline

        with patch("src.pipeline.predict", return_value=None):
            run_pipeline("test", top_k=7)
            # When category is None, query is not expanded
            mock_rag.retrieve.assert_called_once_with("test", 7)

    def test_top_k_none_uses_default(self, mock_rag_cls, mock_rag):
        """None top_k means pipeline uses its default."""
        from src.pipeline import run_pipeline

        with patch("src.pipeline.predict", return_value="Symptoms"):
            run_pipeline("test", category="Symptoms")
            mock_rag.retrieve_by_category.assert_called_once()
            call_args = mock_rag.retrieve_by_category.call_args[0]
            # (question, category, top_k) — top_k is positional arg 2
            assert call_args[2] is None

    def test_top_k_zero(self, mock_rag_cls, mock_rag):
        """top_k=0 is passed through."""
        from src.pipeline import run_pipeline

        with patch("src.pipeline.predict", return_value="Symptoms"):
            run_pipeline("test", top_k=0, category="Symptoms")
            # Query is expanded with category prefix
            mock_rag.retrieve_by_category.assert_called_once_with(
                "symptoms: test", "Symptoms", 0, all_scores=None
            )


# ==============================================================================
# ── run_pipeline — Edge cases ─────────────────────────────────────────────────
# ==============================================================================

class TestRunPipelineEdgeCases:
    """Edge case handling in run_pipeline()."""

    def test_empty_question(self, mock_rag_cls, mock_rag):
        """Empty question string is handled."""
        from src.pipeline import run_pipeline

        with patch("src.pipeline.predict", return_value=None):
            result = run_pipeline("")
            assert isinstance(result, dict)
            assert "answer" in result

    def test_very_long_question(self, mock_rag_cls, mock_rag):
        """Very long question string is passed through."""
        from src.pipeline import run_pipeline

        long_q = "test " * 500
        with patch("src.pipeline.predict", return_value="Treatment"):
            result = run_pipeline(long_q)
            assert result["category"] == "Treatment"

    def test_no_retrieved_results(self, mock_rag_cls, mock_rag):
        """When retrieval returns empty, generate still gets called with empty list."""
        mock_rag.retrieve.return_value = []
        # Override _format_sources to handle empty input
        mock_rag.format_sources = lambda x: []

        from src.pipeline import run_pipeline

        with patch("src.pipeline.predict", return_value=None):
            result = run_pipeline("test")
            assert result["sources"] == []
            assert result["source_details"] == []

    def test_generate_exception(self, mock_rag_cls, mock_rag):
        """When generate raises, the exception propagates."""
        mock_rag.generate.side_effect = RuntimeError("Generation failed")
        from src.pipeline import run_pipeline

        with pytest.raises(RuntimeError, match="Generation failed"):
            with patch("src.pipeline.predict", return_value="General"):
                run_pipeline("test")

    def test_classifier_exception(self, mock_rag_cls, mock_rag):
        """When classifier raises, the exception propagates."""
        from src.pipeline import run_pipeline

        with patch("src.pipeline.predict", side_effect=ValueError("Bad input")):
            with pytest.raises(ValueError, match="Bad input"):
                run_pipeline("test")

    def test_use_classifier_path_with_all_scores(self, mock_rag_cls, mock_rag):
        """When rag._use_classifier is True, predict_with_confidence is used
        and all_scores are passed to retrieve_by_category."""
        mock_rag._use_classifier = True
        mock_rag._classifier = MagicMock()
        mock_rag._classifier.predict_with_confidence.return_value = {
            "category": "Treatment",
            "confidence": 0.85,
            "all_scores": {
                "Treatment": 0.85, "Symptoms": 0.05, "General": 0.05,
                "Diagnosis": 0.03, "Medication": 0.02, "Prevention": 0.0,
            },
        }

        from src.pipeline import run_pipeline

        result = run_pipeline("What is the treatment for hypertension?")
        assert result["category"] == "Treatment"
        mock_rag._classifier.predict_with_confidence.assert_called_once_with(
            "What is the treatment for hypertension?"
        )
        # retrieve_by_category should have been called with all_scores
        mock_rag.retrieve_by_category.assert_called_once()
        call_kwargs = mock_rag.retrieve_by_category.call_args[1]
        assert "all_scores" in call_kwargs
        assert call_kwargs["all_scores"] is not None
        assert call_kwargs["all_scores"]["Treatment"] == 0.85

    def test_multiple_source_details(self, mock_rag_cls, mock_rag):
        """Multiple retrieved results produce multiple source_details."""
        mock_rag.retrieve_by_category.return_value = [
            {"chunk_id": 1, "question": "q1", "category": "Symptoms",
             "distance": 0.1, "relevance_score": 0.9, "excerpt": "e1",
             "reranker_score": 2.0},
            {"chunk_id": 2, "question": "q2", "category": "Symptoms",
             "distance": 0.2, "relevance_score": 0.8, "excerpt": "e2",
             "reranker_score": 1.8},
            {"chunk_id": 3, "question": "q3", "category": "Symptoms",
             "distance": 0.3, "relevance_score": 0.7, "excerpt": "e3",
             "reranker_score": 1.5},
        ]

        from src.pipeline import run_pipeline

        with patch("src.pipeline.predict", return_value="Symptoms"):
            result = run_pipeline("test", category="Symptoms")
            assert len(result["sources"]) == 3
            assert len(result["source_details"]) == 3


# ==============================================================================
# ── Thread safety ─────────────────────────────────────────────────────────────
# ==============================================================================

class TestGetRagThreadSafety:
    """Thread safety of _get_rag() singleton."""

    def test_concurrent_access(self, mock_rag_cls, mock_rag):
        """Multiple threads calling _get_rag() return the same instance."""
        import concurrent.futures
        from src.pipeline import _get_rag

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(lambda: _get_rag()) for _ in range(20)]
            results = [f.result() for f in futures]

        # All calls return the same instance
        assert all(r is results[0] for r in results)
        # build_rag_pipeline was called exactly once
        mock_rag_cls.assert_called_once()
