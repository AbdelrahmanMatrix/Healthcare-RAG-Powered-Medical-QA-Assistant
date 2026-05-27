"""
Unit tests for src/rag/pipeline.py import-time edge cases.

Isolated from test_rag_pipeline_unit.py because these tests modify
sys.modules (delete/reimport the pipeline module), which can corrupt
the module state for other tests that use importlib.reload().
"""

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ==============================================================================
# ── Shared fixtures (minimal set, copied from test_rag_pipeline_unit) ─────────
# ==============================================================================


@pytest.fixture
def mock_index():
    idx = MagicMock()
    idx.ntotal = 100
    return idx


@pytest.fixture
def mock_faiss(mock_index):
    faiss = MagicMock()
    faiss.read_index.return_value = mock_index
    faiss.normalize_L2 = MagicMock()
    return faiss


@pytest.fixture
def mock_encoder():
    enc = MagicMock()
    enc.encode.return_value = np.random.rand(1, 768).astype(np.float32)
    return enc


@pytest.fixture
def mock_st_mod(mock_encoder):
    st = MagicMock()
    st.SentenceTransformer.return_value = mock_encoder
    return st


@pytest.fixture
def mock_openai_mod():
    oa = MagicMock()
    oa.OpenAI = MagicMock(return_value=MagicMock())
    return oa


@pytest.fixture
def mock_bm25():
    bm25 = MagicMock()
    bm25.retrieve.return_value = []
    return bm25


@pytest.fixture
def mock_bm25_mod(mock_bm25):
    bm = MagicMock()
    bm.BM25Retriever = MagicMock(return_value=mock_bm25)
    return bm


@pytest.fixture
def mock_clf_mod():
    return MagicMock()


@pytest.fixture
def mock_df():
    return pd.DataFrame({
        "chunk_id": list(range(100)),
        "question": [f"q{i}" for i in range(100)],
        "answer": [f"a{i}" for i in range(100)],
        "context": [f"c{i}" for i in range(100)],
        "text_chunk": [f"t{i}" for i in range(100)],
        "category": (["Symptoms"] * 20 + ["General"] * 20
                     + ["Treatment"] * 20 + ["Diagnosis"] * 20
                     + ["Medication"] * 20),
    })


@pytest.fixture
def mock_rank_bm25():
    rb = MagicMock()
    rb.BM25Okapi = MagicMock()
    return rb


# ==============================================================================
# ── Stdout encoding fallback ──────────────────────────────────────────────────
# ==============================================================================

class TestStdoutEncodingFallback:
    """Tests for the stdout reconfigure fallback at module level."""

    def test_stdout_reconfigure_fallback_handled(self):
        """When stdout.reconfigure raises, the except block is hit.
        Module-level try/except guards against reconfigure failure on Windows.
        """
        old_mod = sys.modules.get("src.rag.pipeline")

        with patch.object(
            sys.stdout, "reconfigure",
            side_effect=AttributeError("no reconfigure"),
        ):
            # Force module reimport with broken stdout
            if "src.rag.pipeline" in sys.modules:
                del sys.modules["src.rag.pipeline"]
            # Reimport should not crash
            import src.rag.pipeline as rp2  # noqa: F811
            assert hasattr(rp2, "_truncate_words")

        # Restore original module to avoid breaking other tests
        if old_mod is not None:
            sys.modules["src.rag.pipeline"] = old_mod
        else:
            del sys.modules["src.rag.pipeline"]


# ==============================================================================
# ── HAS_TENACITY = False path in _call_groq ───────────────────────────────────
# ==============================================================================

class TestCallGroqWithoutTenacity:
    """Tests that _call_groq works when HAS_TENACITY is False."""

    def test_call_groq_without_tenacity(self, mock_faiss, mock_index,
                                        mock_st_mod, mock_encoder,
                                        mock_openai_mod, mock_bm25_mod,
                                        mock_bm25, mock_clf_mod,
                                        mock_df, mock_rank_bm25):
        """_call_groq works without tenacity (direct call, no retry)."""
        old_mod = sys.modules.get("src.rag.pipeline")

        deps = {
            "faiss": mock_faiss,
            "sentence_transformers": mock_st_mod,
            "openai": mock_openai_mod,
            "rank_bm25": mock_rank_bm25,
            "src.classification.classifier": mock_clf_mod,
            "src.rag.bm25_retriever": mock_bm25_mod,
        }

        # Remove pipeline module so from ... import executes fresh
        if "src.rag.pipeline" in sys.modules:
            del sys.modules["src.rag.pipeline"]

        with patch.dict("sys.modules", deps):
            from src.rag import pipeline as rp
            # Force HAS_TENACITY to False (simulates tenacity not being installed)
            rp.HAS_TENACITY = False

            with (
                patch("builtins.open", MagicMock()),
                patch("pickle.load", return_value=mock_df),
                patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}),
            ):
                pipeline = rp.RAGPipeline(top_k=15, use_reranker=False)

            # Mock the Groq client response
            mock_response = MagicMock()
            mock_response.choices[0].message.content = \
                "  Answer without tenacity  "
            pipeline._groq_client.chat.completions.create.return_value \
                = mock_response

            result = pipeline._call_groq("test prompt")
            assert result == "Answer without tenacity"

        # Restore original module to avoid breaking other tests
        if old_mod is not None:
            sys.modules["src.rag.pipeline"] = old_mod
        else:
            del sys.modules["src.rag.pipeline"]
