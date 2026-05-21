"""
Unit tests for src/rag modules.

Covers:
  - src/rag/bm25_retriever.py   (BM25Retriever)
  - src/rag/embeddings.py       (EmbeddingModel, DEFAULT_MODEL)
  - src/rag/vectorstore.py      (build_index, save_index, load_index,
                                 load_mapping, search)
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

    def test_retrieve_distance_negative(self, retriever):
        """distance is negative of BM25 score (FAISS-consistent)."""
        results = retriever.retrieve("test query", top_k=1)
        assert results[0]["distance"] == -results[0]["bm25_score"]

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
        """Empty query returns top results."""
        results = retriever.retrieve("", top_k=2)
        assert len(results) == 2


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
        mock_st.get_sentence_embedding_dimension.return_value = 768

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
        mock_st.get_sentence_embedding_dimension.return_value = 768
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
        mock_st.get_sentence_embedding_dimension.return_value = 768
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
        mock_st.get_sentence_embedding_dimension.return_value = 768

        return _make_embedding_model(mock_st)

    def test_encode_returns_float32(self, model):
        """encode() returns float32 numpy array."""
        result = model.encode(["hello world"])
        assert result.dtype == np.float32

    def test_encode_batch_size_passed(self):
        """batch_size parameter is passed through to SentenceTransformer.encode."""
        mock_st = MagicMock()
        mock_st.encode.return_value = np.random.rand(2, 768).astype(np.float32)
        mock_st.get_sentence_embedding_dimension.return_value = 768

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
        mock_st.get_sentence_embedding_dimension.return_value = 768

        model = _make_embedding_model(mock_st)
        model.encode_query("test")

        kwargs = mock_st.encode.call_args[1]
        assert kwargs.get("normalize_embeddings") is True


# ==============================================================================
# ── vectorstore.py ────────────────────────────────────────────────────────────
# ==============================================================================

class TestVectorstoreBuildIndex:
    """Tests for vectorstore.build_index()."""

    def test_build_index_flatl2(self):
        """build_index creates an IndexFlatL2 with correct dimension."""
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
            assert "pubmedqa_index_flatl2.faiss" in str(args[0])


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
        assert str(DEFAULT_INDEX_PATH).endswith("pubmedqa_index_flatl2.faiss")

    def test_default_mapping_path_ends_correctly(self):
        """DEFAULT_MAPPING_PATH ends with the expected filename."""
        from src.rag.vectorstore import DEFAULT_MAPPING_PATH
        assert str(DEFAULT_MAPPING_PATH).endswith("chunk_mapping.pkl")
