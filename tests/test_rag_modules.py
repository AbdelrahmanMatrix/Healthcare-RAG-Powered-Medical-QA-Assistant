"""
Unit tests for src/rag modules.

Covers:
  - src/rag/bm25_retriever.py   (BM25Retriever)
  - src/rag/embeddings.py       (EmbeddingModel, DEFAULT_MODEL)
  - src/rag/vectorstore.py      (build_index, save_index, load_index,
                                 load_mapping, search)
  - src/rag/pipeline.py         hybrid retrieval (BM25 + FAISS merge)
"""

import sys
from unittest.mock import patch, MagicMock
import pytest
import numpy as np
import pandas as pd


# ==============================================================================
# ── bm25_retriever.py ─────────────────────────────────────────────────────────
# ==============================================================================

def _make_bm25_module(mock_bm25_class=None):
    """Reload bm25_retriever with rank_bm25 mocked in sys.modules."""
    import importlib as _il

    # Build a fake rank_bm25 module
    mock_rank_bm25 = MagicMock()
    if mock_bm25_class is not None:
        mock_rank_bm25.BM25Okapi = mock_bm25_class
    else:
        mock_rank_bm25.BM25Okapi = MagicMock()

    with patch.dict(sys.modules, {"rank_bm25": mock_rank_bm25}):
        import src.rag.bm25_retriever as bm25_mod
        _il.reload(bm25_mod)
        return bm25_mod


def _make_bm25_retriever(mock_bm25_class, df):
    """Helper: create a BM25Retriever with mocked internals."""
    bm25_mod = _make_bm25_module(mock_bm25_class)
    return bm25_mod.BM25Retriever(df)


class TestBM25RetrieverInit:
    """Tests for BM25Retriever.__init__()."""

    def test_successful_init(self):
        """With rank_bm25 available, builds index successfully."""
        mock_bm25 = MagicMock()
        mock_bm25_class = MagicMock(return_value=mock_bm25)

        mock_df = pd.DataFrame({
            "question": ["q1", "q2"],
            "answer": ["a1", "a2"],
            "context": ["c1", "c2"],
            "text_chunk": ["t1", "t2"],
        })

        retriever = _make_bm25_retriever(mock_bm25_class, mock_df)
        assert retriever.mapping_df is mock_df
        assert retriever.bm25 is mock_bm25
        mock_bm25_class.assert_called_once()

    def test_init_with_category_column(self):
        """When mapping_df has a 'category' column, retrieve includes it."""
        mock_bm25 = MagicMock()
        mock_bm25_class = MagicMock(return_value=mock_bm25)

        mock_df = pd.DataFrame({
            "question": ["q1"], "answer": ["a1"],
            "context": ["c1"], "text_chunk": ["t1"],
            "category": ["Symptoms"],
        })

        retriever = _make_bm25_retriever(mock_bm25_class, mock_df)
        assert retriever.mapping_df is mock_df


class TestBM25RetrieverRetrieve:
    """Tests for BM25Retriever.retrieve()."""

    @pytest.fixture
    def retriever(self):
        """Build a BM25Retriever with mocked BM25 internals."""
        mock_bm25 = MagicMock()
        mock_bm25.get_scores.return_value = np.array([5.0, 3.0, 1.0, 0.0])
        mock_bm25_class = MagicMock(return_value=mock_bm25)

        mock_df = pd.DataFrame({
            "question": ["q1", "q2", "q3", "q4"],
            "answer": ["a1", "a2", "a3", "a4"],
            "context": ["c1", "c2", "c3", "c4"],
            "text_chunk": ["t1", "t2", "t3", "t4"],
        })

        return _make_bm25_retriever(mock_bm25_class, mock_df)

    def test_retrieve_top_k(self, retriever):
        """Returns exactly top_k results."""
        results = retriever.retrieve("test query", top_k=3)
        assert len(results) == 3

    def test_retrieve_default_k(self, retriever):
        """Default top_k is 5, but limited by available docs."""
        results = retriever.retrieve("test query")
        assert len(results) == 4  # only 4 docs in the DataFrame

    def test_retrieve_structure(self, retriever):
        """Each result has the expected keys."""
        results = retriever.retrieve("test query", top_k=1)
        r = results[0]
        assert "chunk_id" in r
        assert "question" in r
        assert "context" in r
        assert "answer" in r
        assert "category" in r
        assert "text_chunk" in r
        assert "distance" in r
        assert "bm25_score" in r

    def test_retrieve_sorted_by_score(self, retriever):
        """Results are sorted by BM25 score descending."""
        results = retriever.retrieve("test query", top_k=4)
        scores = [r["bm25_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_distance_normalized(self, retriever):
        """distance is soft-normalized from BM25 score: 1/(1+score)."""
        results = retriever.retrieve("test query", top_k=1)
        expected = 1.0 / (1.0 + results[0]["bm25_score"])
        assert results[0]["distance"] == pytest.approx(expected)

    def test_retrieve_category_fallback(self):
        """When category column is missing, defaults to 'Unknown'."""
        mock_bm25 = MagicMock()
        mock_bm25.get_scores.return_value = np.array([2.0])
        mock_bm25_class = MagicMock(return_value=mock_bm25)

        mock_df = pd.DataFrame({
            "question": ["q1"], "answer": ["a1"],
            "context": ["c1"], "text_chunk": ["t1"],
        })

        retriever = _make_bm25_retriever(mock_bm25_class, mock_df)
        results = retriever.retrieve("test", top_k=1)
        assert results[0]["category"] == "Unknown"

    def test_retrieve_empty_query(self, retriever):
        """Empty query returns no results (no tokens to match)."""
        results = retriever.retrieve("", top_k=2)
        assert len(results) == 0


class TestBM25RetrieverNotInstalled:
    """When rank_bm25 is not available."""

    def test_rank_bm25_not_installed(self):
        """When rank_bm25 is not importable, raises ImportError."""
        real_import = __builtins__["__import__"]

        def mock_import(name, *args, **kwargs):
            if name == "rank_bm25":
                raise ImportError("No module named 'rank_bm25'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            # Remove cached module for clean reload
            sys.modules.pop("src.rag.bm25_retriever", None)
            import importlib as _il
            import src.rag.bm25_retriever as bm25_mod
            _il.reload(bm25_mod)

            mock_df = pd.DataFrame({
                "question": ["q1"], "answer": ["a1"],
                "context": ["c1"], "text_chunk": ["t1"],
            })
            with pytest.raises(ImportError, match="rank-bm25"):
                bm25_mod.BM25Retriever(mock_df)


# ==============================================================================
# ── embeddings.py ─────────────────────────────────────────────────────────────
# ── sentence_transformers depends on torch (DLL issues on this system).
#    We mock the entire module in sys.modules before any import/reload.
# ==============================================================================

def _make_embeddings_module(mock_st=None):
    """Reload embeddings module with sentence_transformers mocked in sys.modules.

    Args:
        mock_st: A mock SentenceTransformer instance to use during reload.
                 If None, creates a default one.

    Returns:
        The reloaded src.rag.embeddings module.
    """
    import importlib as _il

    if mock_st is None:
        mock_st = MagicMock()
        mock_st.get_embedding_dimension.return_value = 768

    mock_st_mod = MagicMock()
    mock_st_mod.SentenceTransformer.return_value = mock_st

    # Mock at sys.modules level so the import inside embeddings.py gets the mock.
    # This prevents Python from loading the real sentence_transformers (and torch).
    with patch.dict(sys.modules, {"sentence_transformers": mock_st_mod}):
        import src.rag.embeddings as emb_mod
        _il.reload(emb_mod)
        return emb_mod


def _make_embedding_model(mock_st=None):
    """Helper: create an EmbeddingModel instance with mocked internals."""
    emb_mod = _make_embeddings_module(mock_st)
    return emb_mod.EmbeddingModel()


class TestEmbeddingModelInit:
    """Tests for EmbeddingModel.__init__()."""

    def test_default_model_constant(self):
        """DEFAULT_MODEL is the biomedical PubMedBERT model."""
        emb_mod = _make_embeddings_module()
        assert emb_mod.DEFAULT_MODEL == "pritamdeka/S-PubMedBert-MS-MARCO"

    def test_init_creates_model(self):
        """EmbeddingModel loads a SentenceTransformer."""
        mock_st = MagicMock()
        mock_st.get_embedding_dimension.return_value = 768
        mock_st_mod = MagicMock()
        mock_st_mod.SentenceTransformer = MagicMock(return_value=mock_st)

        import importlib as _il
        with patch.dict(sys.modules, {"sentence_transformers": mock_st_mod}):
            sys.modules.pop("src.rag.embeddings", None)
            import src.rag.embeddings as emb_mod
            _il.reload(emb_mod)

            model = emb_mod.EmbeddingModel("test-model")
            assert model.dimension == 768
            mock_st_mod.SentenceTransformer.assert_called_once_with("test-model")

    def test_init_default_model(self):
        """EmbeddingModel defaults to DEFAULT_MODEL."""
        mock_st = MagicMock()
        mock_st.get_embedding_dimension.return_value = 768
        mock_st_mod = MagicMock()
        mock_st_mod.SentenceTransformer = MagicMock(return_value=mock_st)

        import importlib as _il
        with patch.dict(sys.modules, {"sentence_transformers": mock_st_mod}):
            sys.modules.pop("src.rag.embeddings", None)
            import src.rag.embeddings as emb_mod
            _il.reload(emb_mod)

            emb_mod.EmbeddingModel()
            mock_st_mod.SentenceTransformer.assert_called_once_with(emb_mod.DEFAULT_MODEL)


class TestEmbeddingModelEncode:
    """Tests for EmbeddingModel.encode() and encode_query()."""

    @pytest.fixture
    def model(self):
        """Build an EmbeddingModel instance with mocked SentenceTransformer."""
        mock_st = MagicMock()

        def mock_encode(texts, **kwargs):
            n = len(texts) if isinstance(texts, list) else 1
            return np.random.rand(n, 768).astype(np.float32)

        mock_st.encode = mock_encode
        mock_st.get_embedding_dimension.return_value = 768

        return _make_embedding_model(mock_st)

    def test_encode_returns_float32(self, model):
        """encode() returns float32 numpy array."""
        result = model.encode(["hello world"])
        assert result.dtype == np.float32

    def test_encode_batch_size_passed(self):
        """batch_size parameter is passed through to SentenceTransformer.encode."""
        mock_st = MagicMock()
        mock_st.encode.return_value = np.random.rand(2, 768).astype(np.float32)
        mock_st.get_embedding_dimension.return_value = 768

        model = _make_embedding_model(mock_st)
        model.encode(["a", "b"], batch_size=64)

        kwargs = mock_st.encode.call_args[1]
        assert kwargs["batch_size"] == 64

    def test_encode_multiple_texts(self, model):
        """encode() handles multiple texts."""
        result = model.encode(["a", "b", "c"])
        assert result.shape[0] == 3

    def test_encode_query_returns_2d_array(self, model):
        """encode_query() returns a 2D numpy array (1, dim)."""
        result = model.encode_query("test query")
        assert result.ndim == 2
        assert result.shape[0] == 1
        assert result.shape[1] == 768
        assert result.dtype == np.float32

    def test_encode_query_normalized(self):
        """encode_query passes normalize_embeddings=True."""
        mock_st = MagicMock()
        mock_st.encode.return_value = np.random.rand(1, 768).astype(np.float32)
        mock_st.get_embedding_dimension.return_value = 768

        model = _make_embedding_model(mock_st)
        model.encode_query("test")

        kwargs = mock_st.encode.call_args[1]
        assert kwargs.get("normalize_embeddings") is True


# ==============================================================================
# ── vectorstore.py ────────────────────────────────────────────────────────────
# ==============================================================================

class TestVectorstoreBuildIndex:
    """Tests for vectorstore.build_index()."""

    def test_build_index_flatip(self):
        """build_index creates an IndexFlatIP with correct dimension."""
        from src.rag.vectorstore import build_index

        embeddings = np.random.rand(10, 768).astype(np.float32)
        index = build_index(embeddings)
        assert index.d == 768
        assert index.ntotal == 10

    def test_build_index_small(self):
        """build_index works with a single embedding."""
        from src.rag.vectorstore import build_index

        embeddings = np.random.rand(1, 128).astype(np.float32)
        index = build_index(embeddings)
        assert index.ntotal == 1


class TestVectorstoreSearch:
    """Tests for vectorstore.search()."""

    def test_search_returns_distances_and_indices(self):
        """search returns (distances, indices) tuple."""
        from src.rag.vectorstore import build_index, search

        embeddings = np.random.rand(50, 64).astype(np.float32)
        index = build_index(embeddings)

        query = np.random.rand(1, 64).astype(np.float32)
        distances, indices = search(index, query, k=3)
        assert distances.shape == (1, 3)
        assert indices.shape == (1, 3)

    def test_search_k_param(self):
        """search respects the k parameter."""
        from src.rag.vectorstore import build_index, search

        embeddings = np.random.rand(50, 64).astype(np.float32)
        index = build_index(embeddings)

        query = np.random.rand(1, 64).astype(np.float32)
        distances, indices = search(index, query, k=10)
        assert distances.shape == (1, 10)

    def test_search_default_k(self):
        """search defaults to k=5."""
        from src.rag.vectorstore import build_index, search

        embeddings = np.random.rand(50, 64).astype(np.float32)
        index = build_index(embeddings)

        query = np.random.rand(1, 64).astype(np.float32)
        distances, indices = search(index, query)
        assert distances.shape == (1, 5)


class TestVectorstoreIO:
    """Tests for vectorstore.save_index() and load_index()."""

    def test_save_and_load_index(self, tmp_path):
        """Save then load returns an index with same number of vectors."""
        from src.rag.vectorstore import build_index, save_index, load_index

        embeddings = np.random.rand(10, 64).astype(np.float32)
        original = build_index(embeddings)

        index_path = str(tmp_path / "test_index.faiss")
        save_index(original, index_path)

        loaded = load_index(index_path)
        assert loaded.ntotal == 10
        assert loaded.d == 64

    def test_load_index_none_path_defaults(self):
        """load_index with path=None uses DEFAULT_INDEX_PATH."""
        with patch("src.rag.vectorstore.faiss.read_index") as mock_read:
            mock_read.return_value = MagicMock()
            from src.rag.vectorstore import load_index

            load_index()
            mock_read.assert_called_once()
            args, _ = mock_read.call_args
            assert "pubmedqa_index_flatip.faiss" in str(args[0])


class TestVectorstoreMapping:
    """Tests for vectorstore.load_mapping()."""

    def test_load_mapping_loads_dataframe(self, tmp_path):
        """load_mapping loads a pickle-saved DataFrame."""
        from src.rag.vectorstore import load_mapping
        import pickle

        df = pd.DataFrame({
            "chunk_id": [0, 1],
            "question": ["q1", "q2"],
            "context": ["c1", "c2"],
            "text_chunk": ["t1", "t2"],
        })
        mapping_path = tmp_path / "chunk_mapping.pkl"
        with open(mapping_path, "wb") as f:
            pickle.dump(df, f)

        loaded = load_mapping(str(mapping_path))
        assert isinstance(loaded, pd.DataFrame)
        assert len(loaded) == 2
        assert list(loaded["question"]) == ["q1", "q2"]

    def test_load_mapping_none_path_defaults(self):
        """load_mapping with path=None uses DEFAULT_MAPPING_PATH."""
        with patch("builtins.open", MagicMock()) as mock_open, \
             patch("pickle.load", return_value=pd.DataFrame()):
            mock_open.return_value.__enter__.return_value = MagicMock()
            from src.rag.vectorstore import load_mapping

            load_mapping()
            args, _ = mock_open.call_args
            assert "chunk_mapping.pkl" in str(args[0])


class TestVectorstoreDefaultPaths:
    """Tests for vectorstore default path constants."""

    def test_default_index_path_ends_correctly(self):
        """DEFAULT_INDEX_PATH ends with the expected filename."""
        from src.rag.vectorstore import DEFAULT_INDEX_PATH
        assert str(DEFAULT_INDEX_PATH).endswith("pubmedqa_index_flatip.faiss")

    def test_default_mapping_path_ends_correctly(self):
        """DEFAULT_MAPPING_PATH ends with the expected filename."""
        from src.rag.vectorstore import DEFAULT_MAPPING_PATH
        assert str(DEFAULT_MAPPING_PATH).endswith("chunk_mapping.pkl")


# ==============================================================================
# ── pipeline.py category expansion fallback ───────────────────────────────────
# ==============================================================================

class TestRAGPipelineCategoryExpansion:
    """Tests for retrieve_by_category() expansion factor fallback logic.

    When a category is not found in CATEGORY_EXPANSION (e.g. misspelled or
    unknown), retrieve_by_category must default to 3x expansion.
    """

    def test_category_expansion_constant_fallback(self):
        """Unknown/misspelled category falls back to 3x via CATEGORY_EXPANSION.get()."""
        from src.rag.pipeline import CATEGORY_EXPANSION

        # Known categories return their specific factors unchanged
        assert CATEGORY_EXPANSION["Medication"] == 2
        assert CATEGORY_EXPANSION["Treatment"] == 2
        assert CATEGORY_EXPANSION["Diagnosis"] == 3
        assert CATEGORY_EXPANSION["General"] == 3
        assert CATEGORY_EXPANSION["Prevention"] == 4
        assert CATEGORY_EXPANSION["Symptoms"] == 5

        # Unknown categories fall back to 3 via .get(category, 3)
        assert CATEGORY_EXPANSION.get("Cardiology", 3) == 3
        assert CATEGORY_EXPANSION.get("Symtoms", 3) == 3     # misspelled
        assert CATEGORY_EXPANSION.get("UnknownCategory", 3) == 3
        assert CATEGORY_EXPANSION.get("", 3) == 3            # empty string

    def test_retrieve_by_category_unknown_uses_3x_expansion(self):
        """retrieve_by_category uses 3x FAISS search_k for unknown/misspelled categories."""
        import os
        import numpy as np
        import pandas as pd
        import sys
        import importlib as _il

        # Build all mocks
        mock_index = MagicMock()
        mock_index.ntotal = 500

        mock_faiss = MagicMock()
        mock_faiss.read_index.return_value = mock_index
        mock_faiss.normalize_L2 = MagicMock()

        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = np.random.rand(1, 768).astype(np.float32)

        mock_st_mod = MagicMock()
        mock_st_mod.SentenceTransformer.return_value = mock_encoder

        mock_openai_mod = MagicMock()
        mock_openai_mod.OpenAI = MagicMock(return_value=MagicMock())

        mock_df = pd.DataFrame({
            "chunk_id": list(range(100)),
            "question": [f"q{i}" for i in range(100)],
            "answer": [f"a{i}" for i in range(100)],
            "context": [f"c{i}" for i in range(100)],
            "text_chunk": [f"t{i}" for i in range(100)],
            "category": ["General"] * 100,
        })

        # Mock classifier module to prevent torch DLL access violations on Windows
        mock_clf_mod = MagicMock()

        # Reload pipeline module with mocked dependencies
        with patch.dict("sys.modules", {
            "faiss": mock_faiss,
            "sentence_transformers": mock_st_mod,
            "openai": mock_openai_mod,
            "src.classification.classifier": mock_clf_mod,
        }):
            from src.rag import pipeline as rp
            _il.reload(rp)

            with (
                patch("builtins.open", MagicMock()),
                patch("pickle.load", return_value=mock_df),
                patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}),
            ):
                pipeline = rp.RAGPipeline(top_k=15, use_reranker=False)

        # Wire up mock index.search to capture search_k
        search_calls = []

        def search_side_effect(query, k):
            search_calls.append(k)
            return (
                np.random.rand(1, k).astype(np.float32),
                np.random.randint(0, 100, size=(1, k)).astype(np.int64),
            )

        mock_index.search = MagicMock(side_effect=search_side_effect)

        # Test: unknown category -> factor=3 -> search_k = 15x3 = 45
        pipeline.retrieve_by_category("test query", "UnknownCategory")
        assert search_calls[-1] == 45, (
            f"Unknown category: expected search_k=45 (15x3), got {search_calls[-1]}"
        )

        # Test: misspelled category -> factor=3 -> search_k = 45
        pipeline.retrieve_by_category("test query", "Symtoms")
        assert search_calls[-1] == 45, (
            f"Misspelled category: expected search_k=45 (15x3), got {search_calls[-1]}"
        )

        # Test: known category uses its specific factor
        pipeline.retrieve_by_category("test query", "Symptoms")
        assert search_calls[-1] == 75, (
            f"Symptoms: expected search_k=75 (15x5), got {search_calls[-1]}"
        )

        # Test: another known category uses its specific factor
        pipeline.retrieve_by_category("test query", "General")
        assert search_calls[-1] == 45, (
            f"General: expected search_k=45 (15x3), got {search_calls[-1]}"
        )


# ==============================================================================
# ── Hybrid retrieval integration (BM25 + FAISS) ───────────────────────────────
# ==============================================================================

class _HybridPipelineBuilder:
    """Helper to build a RAGPipeline with fully mocked internals for hybrid retrieval tests.

    Provides a reusable pipeline that exercises the BM25 + FAISS merge logic
    in retrieve() and retrieve_by_category() without loading real models.
    """

    def __init__(self, mock_df=None, use_bm25=True, use_reranker=False, top_k=15, bm25_threshold=12.0):
        import os
        import numpy as np
        import sys
        import importlib as _il

        self._np = np

        # Build mock FAISS index
        mock_index = MagicMock()
        mock_index.ntotal = len(mock_df) if mock_df is not None else 100

        mock_faiss = MagicMock()
        mock_faiss.read_index.return_value = mock_index
        mock_faiss.normalize_L2 = MagicMock()

        # Build mock encoder
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = np.random.rand(1, 768).astype(np.float32)

        mock_st_mod = MagicMock()
        mock_st_mod.SentenceTransformer.return_value = mock_encoder

        # Build mock BM25 retriever
        if use_bm25:
            mock_bm25 = MagicMock()
            mock_bm25_mod = MagicMock()
            mock_bm25_mod.BM25Retriever = MagicMock(return_value=mock_bm25)
            mock_rank_bm25 = MagicMock()
            mock_rank_bm25.BM25Okapi = MagicMock()
        else:
            mock_bm25 = None
            # When BM25 is unavailable, we do NOT mock src.rag.bm25_retriever
            # in sys.modules, so the real import runs.
            # We also mock rank_bm25 to lack BM25Okapi, ensuring
            # HAS_BM25 = False inside bm25_retriever.py, which causes
            # BM25Retriever.__init__ to raise ImportError -- exercising
            # the pipeline's except ImportError fallback path.
            mock_bm25_mod = None  # Will NOT be placed in sys.modules
            mock_rank_bm25 = MagicMock(spec=[])  # No BM25Okapi attribute

        # Mock openai for Groq client init
        mock_openai_mod = MagicMock()
        mock_openai_mod.OpenAI = MagicMock(return_value=MagicMock())

        # Default DataFrame for chunk mapping
        if mock_df is None:
            mock_df = pd.DataFrame({
                "chunk_id": list(range(100)),
                "question": [f"q{i}" for i in range(100)],
                "answer": [f"a{i}" for i in range(100)],
                "context": [f"c{i}" for i in range(100)],
                "text_chunk": [f"t{i}" for i in range(100)],
                "category": ["General"] * 100,
            })

        self._mock_df = mock_df
        self._mock_bm25 = mock_bm25
        self._mock_index = mock_index
        self._search_calls = []

        # Mock classifier module to prevent torch DLL access violations on Windows
        mock_clf_mod = MagicMock()

        # Reload pipeline with mocked deps
        with patch.dict("sys.modules", {
            "faiss": mock_faiss,
            "sentence_transformers": mock_st_mod,
            "openai": mock_openai_mod,
            "src.rag.bm25_retriever": mock_bm25_mod,
            "rank_bm25": mock_rank_bm25,
            "src.classification.classifier": mock_clf_mod,
        }):
            from src.rag import pipeline as rp
            _il.reload(rp)

            with (
                patch("builtins.open", MagicMock()),
                patch("pickle.load", return_value=mock_df),
                patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}),
            ):
                self._pipeline = rp.RAGPipeline(top_k=top_k, use_reranker=use_reranker)

        # Wire the BM25 threshold if provided (to avoid depending on settings.py)
        if bm25_threshold is not None:
            self._pipeline._bm25_threshold = bm25_threshold

    @property
    def pipeline(self):
        return self._pipeline

    @property
    def mock_bm25(self):
        return self._mock_bm25

    @property
    def mock_index(self):
        return self._mock_index

    @property
    def mock_df(self):
        return self._mock_df

    @property
    def search_calls(self):
        return self._search_calls

    def wire_search(self, num_rows=100):
        """Wire mock_index.search to return deterministic indices and capture search_k."""
        self._search_calls = []

        def search_side_effect(query, k):
            self._search_calls.append(k)
            # Return deterministic indices 0..k-1 and distances ~0.5
            return (
                np.full((1, k), 0.5, dtype=np.float32),
                np.arange(k, dtype=np.int64).reshape(1, k),
            )

        self._mock_index.search = MagicMock(side_effect=search_side_effect)


class TestHybridRetrievalBasic:
    """Tests for the BM25 + FAISS hybrid merge in retrieve()."""

    def _make_builder(self, bm25_scores=None, bm25_threshold=12.0, top_k=15, use_bm25=True):
        """Build pipeline + wire BM25 scores."""
        builder = _HybridPipelineBuilder(
            use_bm25=use_bm25, top_k=top_k, bm25_threshold=bm25_threshold
        )
        builder.wire_search()

        if use_bm25:
            # Configure BM25 mock
            if bm25_scores is not None:
                n = len(bm25_scores)
                builder.mock_bm25.retrieve.return_value = [
                    {
                        "chunk_id": int(i),
                        "question": f"bm25_q{i}",
                        "context": f"bm25_c{i}",
                        "answer": f"bm25_a{i}",
                        "category": "General",
                        "text_chunk": f"bm25_t{i}",
                        "distance": 1.0 / (1.0 + bm25_scores[i]),
                        "bm25_score": bm25_scores[i],
                    }
                    for i in range(n)
                ]

        return builder

    def test_bm25_above_threshold_prepended(self):
        """BM25 results above threshold appear before FAISS results."""
        bm25_scores = [15.0, 13.0]  # both above 12.0 threshold
        builder = self._make_builder(bm25_scores=bm25_scores, bm25_threshold=12.0, top_k=10)
        pipeline = builder.pipeline

        results = pipeline.retrieve("test query", top_k=10)

        # First 2 results should be from BM25 (above threshold)
        assert len(results) > 0
        # BM25 results should be first in the merged list
        for i in range(2):
            assert results[i]["question"] == f"bm25_q{i}"

    def test_bm25_below_threshold_excluded(self):
        """BM25 results below threshold are excluded from merge."""
        bm25_scores = [8.0, 5.0]  # both below 12.0 threshold
        builder = self._make_builder(bm25_scores=bm25_scores, bm25_threshold=12.0, top_k=10)
        pipeline = builder.pipeline

        results = pipeline.retrieve("test query", top_k=10)

        # No BM25 results should appear (all below threshold)
        for r in results:
            assert "bm25_q" not in r.get("question", "")

    def test_mixed_bm25_scores(self):
        """Mix of above/below threshold: only above-threshold BM25 results are prepended."""
        bm25_scores = [15.0, 8.0, 14.0, 5.0, 13.0]
        builder = self._make_builder(bm25_scores=bm25_scores, bm25_threshold=12.0, top_k=10)
        pipeline = builder.pipeline

        results = pipeline.retrieve("test query", top_k=10)

        # BM25 results above threshold (15.0, 14.0, 13.0) should be at indices 0, 1, 2
        bm25_questions_in_results = [r["question"] for r in results if r["question"].startswith("bm25_q")]
        assert len(bm25_questions_in_results) == 3
        assert bm25_questions_in_results == ["bm25_q0", "bm25_q2", "bm25_q4"]

    def test_deduplication(self):
        """Same chunk_id from BM25 and FAISS is not duplicated."""
        bm25_scores = [15.0, 14.0]
        builder = _HybridPipelineBuilder(
            use_bm25=True, top_k=10, bm25_threshold=12.0
        )
        builder.wire_search()

        # BM25 returns chunk_ids 0 and 1 (same as FAISS top-2)
        builder.mock_bm25.retrieve.return_value = [
            {
                "chunk_id": int(i),
                "question": f"bm25_q{i}",
                "context": f"bm25_c{i}",
                "answer": f"bm25_a{i}",
                "category": "General",
                "text_chunk": f"bm25_t{i}",
                "distance": 1.0 / (1.0 + 15.0),
                "bm25_score": 15.0,
            }
            for i in range(2)
        ]
        pipeline = builder.pipeline

        results = pipeline.retrieve("test query", top_k=10)

        # Collect all chunk_ids
        chunk_ids = [r["chunk_id"] for r in results]
        # No duplicates
        assert len(chunk_ids) == len(set(chunk_ids))
        # BM25 results come first, then FAISS fills remaining unique
        assert chunk_ids[0] == 0
        assert chunk_ids[1] == 1

    def test_bm25_disabled_faiss_only(self):
        """When BM25 is unavailable, falls back to FAISS-only retrieval."""
        builder = self._make_builder(use_bm25=False, top_k=10)
        pipeline = builder.pipeline

        results = pipeline.retrieve("test query", top_k=10)

        # All results should be FAISS (no bm25_ prefix in question)
        for r in results:
            assert not r.get("question", "").startswith("bm25_")
        # Should have exactly top_k results (FAISS returns 10)
        assert len(results) == 10

    def test_top_k_respected(self):
        """retrieve() caps results at top_k."""
        bm25_scores = [15.0, 14.0, 13.0, 12.5]
        builder = self._make_builder(bm25_scores=bm25_scores, bm25_threshold=12.0, top_k=3)
        pipeline = builder.pipeline

        results = pipeline.retrieve("test query", top_k=3)

        assert len(results) <= 3

    def test_empty_query_returns_empty(self):
        """Empty query returns empty list (no tokens to match)."""
        builder = self._make_builder(use_bm25=True, top_k=10)
        pipeline = builder.pipeline

        # BM25 with empty query returns empty
        builder.mock_bm25.retrieve.return_value = []

        # FAISS on empty query still returns something, but BM25 shouldn't add
        results = pipeline.retrieve("", top_k=10)
        assert isinstance(results, list)

    def test_retrieve_zero_k_returns_empty(self):
        """retrieve with top_k=0 returns empty list."""
        bm25_scores = [15.0]
        builder = self._make_builder(bm25_scores=bm25_scores, bm25_threshold=12.0, top_k=0)
        pipeline = builder.pipeline

        results = pipeline.retrieve("test query", top_k=0)
        assert results == []

    def test_retrieve_structure_hybrid(self):
        """Hybrid results have the expected dict keys."""
        bm25_scores = [15.0, 14.0]
        builder = self._make_builder(bm25_scores=bm25_scores, bm25_threshold=12.0, top_k=5)
        pipeline = builder.pipeline

        results = pipeline.retrieve("test query", top_k=5)

        for r in results:
            assert "chunk_id" in r
            assert "question" in r
            assert "context" in r
            assert "answer" in r
            assert "category" in r
            assert "distance" in r

    def test_reranker_integration(self):
        """When reranker is enabled, results have reranker_score key."""
        bm25_scores = [15.0]
        builder = _HybridPipelineBuilder(
            use_bm25=True, use_reranker=True, top_k=5, bm25_threshold=12.0
        )
        builder.wire_search()
        builder.mock_bm25.retrieve.return_value = [
            {
                "chunk_id": 0,
                "question": "bm25_q0",
                "context": "bm25_c0",
                "answer": "bm25_a0",
                "category": "General",
                "text_chunk": "bm25_t0",
                "distance": 0.1,
                "bm25_score": 15.0,
            }
        ]

        # Mock the reranker's predict method
        pipeline = builder.pipeline
        pipeline.reranker = MagicMock()
        pipeline.reranker.predict.return_value = [0.95, 0.85, 0.75, 0.65, 0.55]

        results = pipeline.retrieve("test query", top_k=5)

        for r in results:
            assert "reranker_score" in r
        # Results should be sorted by reranker_score descending
        scores = [r["reranker_score"] for r in results]
        assert scores == sorted(scores, reverse=True)


class TestHybridRetrievalByCategory:
    """Tests for category-routed hybrid retrieval with BM25."""

    def test_retrieve_by_category_with_bm25(self):
        """retrieve_by_category merges BM25 + FAISS with category expansion."""
        builder = _HybridPipelineBuilder(
            use_bm25=True, top_k=15, bm25_threshold=12.0
        )
        builder.wire_search()

        # BM25 returns 2 results above threshold
        builder.mock_bm25.retrieve.return_value = [
            {
                "chunk_id": 200,
                "question": "bm25_q200",
                "context": "bm25_c200",
                "answer": "bm25_a200",
                "category": "Symptoms",
                "text_chunk": "bm25_t200",
                "distance": 0.1,
                "bm25_score": 15.0,
            },
            {
                "chunk_id": 201,
                "question": "bm25_q201",
                "context": "bm25_c201",
                "answer": "bm25_a201",
                "category": "Symptoms",
                "text_chunk": "bm25_t201",
                "distance": 0.2,
                "bm25_score": 14.0,
            },
        ]

        pipeline = builder.pipeline

        results = pipeline.retrieve_by_category("test query", "Symptoms")

        # BM25 results should be included
        questions = [r["question"] for r in results]
        assert "bm25_q200" in questions
        assert "bm25_q201" in questions
        # FAISS results fill the rest
        assert len(results) > 2

    def test_retrieve_by_category_no_bm25(self):
        """retrieve_by_category works without BM25 (FAISS-only + category expansion)."""
        builder = _HybridPipelineBuilder(
            use_bm25=False, top_k=15
        )
        builder.wire_search()

        pipeline = builder.pipeline

        results = pipeline.retrieve_by_category("test query", "Symptoms")
        assert len(results) > 0
        # All results should be FAISS (no bm25_ prefix)
        for r in results:
            assert not r.get("question", "").startswith("bm25_")

    def test_retrieve_by_category_bm25_below_threshold(self):
        """BM25 results below threshold don't appear in category-routed results."""
        builder = _HybridPipelineBuilder(
            use_bm25=True, top_k=15, bm25_threshold=12.0
        )
        builder.wire_search()

        # BM25 scores below threshold
        builder.mock_bm25.retrieve.return_value = [
            {
                "chunk_id": 300,
                "question": "bm25_q300",
                "context": "bm25_c300",
                "answer": "bm25_a300",
                "category": "General",
                "text_chunk": "bm25_t300",
                "distance": 0.5,
                "bm25_score": 5.0,
            }
        ]

        pipeline = builder.pipeline

        results = pipeline.retrieve_by_category("test query", "General")
        questions = [r["question"] for r in results]
        assert "bm25_q300" not in questions

    def test_retrieve_by_category_expansion_factor(self):
        """Category expansion factor controls FAISS search_k in hybrid path."""
        builder = _HybridPipelineBuilder(
            use_bm25=True, top_k=15, bm25_threshold=12.0
        )
        builder.wire_search()

        # BM25 returns nothing
        builder.mock_bm25.retrieve.return_value = []

        pipeline = builder.pipeline

        # Symptoms has expansion factor 5 -> search_k = 15 * 5 = 75
        pipeline.retrieve_by_category("test query", "Symptoms")
        assert builder.search_calls[-1] == 75

        # Medication has expansion factor 2 -> search_k = 15 * 2 = 30
        pipeline.retrieve_by_category("test query", "Medication")
        assert builder.search_calls[-1] == 30

    def test_retrieve_by_category_continuous_scoring(self):
        """All_scores dict enables continuous category scoring."""
        import os
        import sys
        import importlib as _il
        import numpy as np
        import pandas as pd
        from unittest.mock import MagicMock, patch

        mock_index = MagicMock()
        mock_index.ntotal = 100

        mock_faiss = MagicMock()
        mock_faiss.read_index.return_value = mock_index
        mock_faiss.normalize_L2 = MagicMock()

        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = np.random.rand(1, 768).astype(np.float32)

        mock_st_mod = MagicMock()
        mock_st_mod.SentenceTransformer.return_value = mock_encoder

        mock_bm25 = MagicMock()
        mock_bm25.retrieve.return_value = []

        mock_bm25_mod = MagicMock()
        mock_bm25_mod.BM25Retriever = MagicMock(return_value=mock_bm25)

        mock_openai_mod = MagicMock()
        mock_openai_mod.OpenAI = MagicMock(return_value=MagicMock())

        # DataFrame with diverse categories
        mock_df = pd.DataFrame({
            "chunk_id": list(range(100)),
            "question": [f"q{i}" for i in range(100)],
            "answer": [f"a{i}" for i in range(100)],
            "context": [f"c{i}" for i in range(100)],
            "text_chunk": [f"t{i}" for i in range(100)],
            "category": ["Symptoms"] * 20 + ["General"] * 20 + ["Treatment"] * 20 + ["Diagnosis"] * 20 + ["Medication"] * 20,
        })

        with patch.dict("sys.modules", {
            "faiss": mock_faiss,
            "sentence_transformers": mock_st_mod,
            "openai": mock_openai_mod,
            "src.rag.bm25_retriever": mock_bm25_mod,
            "src.classification.classifier": MagicMock(),
        }):
            from src.rag import pipeline as rp
            _il.reload(rp)

            with (
                patch("builtins.open", MagicMock()),
                patch("pickle.load", return_value=mock_df),
                patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}),
            ):
                pipeline = rp.RAGPipeline(top_k=15, use_reranker=False)

        # Wire index.search with deterministic categories
        # Return indices 0..search_k-1 sequentially, so each row's category is known
        def search_side_effect(query, k):
            return (
                np.full((1, k), 0.5, dtype=np.float32),
                np.arange(k, dtype=np.int64).reshape(1, k),
            )

        mock_index.search = MagicMock(side_effect=search_side_effect)

        # Provide all_scores: Symptoms gets high probability
        all_scores = {
            "Symptoms": 0.85,
            "General": 0.05,
            "Treatment": 0.05,
            "Diagnosis": 0.03,
            "Medication": 0.02,
        }

        results = pipeline.retrieve_by_category(
            "test query", "Symptoms", all_scores=all_scores
        )

        # Results should have category_score key
        for r in results:
            assert "category_score" in r

        # Symptoms chunks should be scored highest
        symptoms_results = [r for r in results if r["category"] == "Symptoms"]
        non_symptoms = [r for r in results if r["category"] != "Symptoms"]
        if symptoms_results and non_symptoms:
            min_symptoms_score = min(r["category_score"] for r in symptoms_results)
            max_non_score = max(r["category_score"] for r in non_symptoms)
            assert min_symptoms_score >= max_non_score, (
                "Symptoms chunks should have higher or equal category_score "
                "than non-Symptoms chunks"
            )

    def test_retrieve_by_category_bm25_threshold_hybrid(self):
        """BM25 threshold works correctly in category-routed hybrid path."""
        builder = _HybridPipelineBuilder(
            use_bm25=True, top_k=15, bm25_threshold=12.0
        )
        builder.wire_search()

        # One BM25 result above threshold, one below
        builder.mock_bm25.retrieve.return_value = [
            {
                "chunk_id": 400,
                "question": "bm25_above",
                "context": "bm25_c400",
                "answer": "bm25_a400",
                "category": "General",
                "text_chunk": "bm25_t400",
                "distance": 0.1,
                "bm25_score": 15.0,
            },
            {
                "chunk_id": 401,
                "question": "bm25_below",
                "context": "bm25_c401",
                "answer": "bm25_a401",
                "category": "General",
                "text_chunk": "bm25_t401",
                "distance": 0.5,
                "bm25_score": 5.0,
            },
        ]

        pipeline = builder.pipeline

        results = pipeline.retrieve_by_category("test query", "General")
        questions = [r["question"] for r in results]

        assert "bm25_above" in questions
        assert "bm25_below" not in questions


class TestHybridRetrievalEdgeCases:
    """Edge cases for hybrid retrieval."""

    def test_all_bm25_duplicated_in_faiss(self):
        """When all BM25 results are already in FAISS, no duplication occurs."""
        builder = _HybridPipelineBuilder(
            use_bm25=True, top_k=10, bm25_threshold=12.0
        )
        builder.wire_search()

        # BM25 returns chunks 0-4 (same as FAISS top-5 due to wire_search returning 0..k-1)
        builder.mock_bm25.retrieve.return_value = [
            {
                "chunk_id": i,
                "question": f"bm25_q{i}",
                "context": f"bm25_c{i}",
                "answer": f"bm25_a{i}",
                "category": "General",
                "text_chunk": f"bm25_t{i}",
                "distance": 0.1,
                "bm25_score": 15.0,
            }
            for i in range(5)
        ]

        pipeline = builder.pipeline

        results = pipeline.retrieve("test query", top_k=10)

        # All chunk_ids should be unique
        chunk_ids = [r["chunk_id"] for r in results]
        assert len(chunk_ids) == len(set(chunk_ids))
        # Total should be 10 (no duplicates means 5 BM25 + 5 FAISS fill to top_k)
        assert len(results) == 10
        # BM25 results should be first (prepended)
        for i in range(5):
            assert results[i]["question"].startswith("bm25_")

    def test_empty_mapping_df(self):
        """Empty mapping DataFrame is handled gracefully."""
        builder = _HybridPipelineBuilder(
            mock_df=pd.DataFrame(columns=[
                "chunk_id", "question", "answer", "context", "text_chunk", "category"
            ]),
            use_bm25=True, top_k=5, bm25_threshold=12.0
        )

        # BM25 returns empty
        builder.mock_bm25.retrieve.return_value = []

        # index.ntotal = 0
        builder.mock_index.ntotal = 0

        pipeline = builder.pipeline
        results = pipeline.retrieve("test", top_k=5)
        assert results == []

    def test_bm25_returns_empty_faiss_only(self):
        """When BM25 returns empty list, FAISS-only results are used."""
        builder = _HybridPipelineBuilder(
            use_bm25=True, top_k=10, bm25_threshold=12.0
        )
        builder.wire_search()

        # BM25 returns empty (e.g., empty query tokens)
        builder.mock_bm25.retrieve.return_value = []

        pipeline = builder.pipeline

        results = pipeline.retrieve("test query", top_k=10)
        assert len(results) == 10
        # All results should be from FAISS
        for r in results:
            assert not r.get("question", "").startswith("bm25_")

    def test_top_k_larger_than_index(self):
        """top_k larger than index size is capped to ntotal."""
        builder = _HybridPipelineBuilder(
            use_bm25=True, top_k=20, bm25_threshold=12.0
        )

        # Override index ntotal to be smaller than top_k
        builder.mock_index.ntotal = 5
        builder.wire_search()
        builder.mock_bm25.retrieve.return_value = [
            {
                "chunk_id": 0,
                "question": "bm25_q0",
                "context": "bm25_c0",
                "answer": "bm25_a0",
                "category": "General",
                "text_chunk": "bm25_t0",
                "distance": 0.1,
                "bm25_score": 15.0,
            }
        ]

        pipeline = builder.pipeline
        results = pipeline.retrieve("test query", top_k=20)
        # Should be capped to index ntotal
        assert len(results) <= 5

    def test_all_bm25_scores_exactly_at_threshold(self):
        """BM25 scores exactly at the threshold are included."""
        builder = _HybridPipelineBuilder(
            use_bm25=True, top_k=5, bm25_threshold=12.0
        )
        builder.wire_search()

        builder.mock_bm25.retrieve.return_value = [
            {
                "chunk_id": 500,
                "question": "bm25_at_threshold",
                "context": "bm25_c500",
                "answer": "bm25_a500",
                "category": "General",
                "text_chunk": "bm25_t500",
                "distance": 1.0 / (1.0 + 12.0),
                "bm25_score": 12.0,  # exactly at threshold
            }
        ]

        pipeline = builder.pipeline
        results = pipeline.retrieve("test query", top_k=5)

        # 12.0 > 12.0 is False, so it should NOT be included
        questions = [r["question"] for r in results]
        assert "bm25_at_threshold" not in questions
