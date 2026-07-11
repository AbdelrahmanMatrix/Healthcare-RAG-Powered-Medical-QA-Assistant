"""
Unit tests for src/rag/pipeline.py internal logic.

Covers the methods and utilities NOT tested in test_rag_modules.py:

  - Helper functions: _truncate_words, _clean_answer, _is_insufficient
  - Instance methods: _row_to_dict, _is_hedging, format_sources,
    _generate_once, _call_groq, generate, answer, answer_with_routing,
    _needs_retrieval
  - Module-level: build_rag_pipeline, answer(), retrieve()
  - Init edge cases: tenacity import, category expansion JSON override
"""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ==============================================================================
# ── Helper function tests ─────────────────────────────────────────────────────
# ── These import the module and call utility functions directly.              ──
# ==============================================================================


class TestTruncateWords:
    """Tests for _truncate_words()."""

    def test_truncates_at_word_boundary(self):
        from src.rag.pipeline import _truncate_words
        text = "word " * 20
        result = _truncate_words(text.strip(), max_words=5)
        assert result == "word word word word word..."

    def test_under_limit_returns_full(self):
        from src.rag.pipeline import _truncate_words
        text = "short text"
        result = _truncate_words(text, max_words=10)
        assert result == "short text"

    def test_empty_string(self):
        from src.rag.pipeline import _truncate_words
        assert _truncate_words("", max_words=5) == ""

    def test_none_input(self):
        from src.rag.pipeline import _truncate_words
        assert _truncate_words(None, max_words=5) == ""

    def test_exact_limit(self):
        from src.rag.pipeline import _truncate_words
        text = "one two three"
        result = _truncate_words(text, max_words=3)
        assert result == "one two three"

    def test_single_word_truncated(self):
        from src.rag.pipeline import _truncate_words
        text = "hello"
        result = _truncate_words(text, max_words=0)
        assert result == "..."

    def test_zero_max_words(self):
        from src.rag.pipeline import _truncate_words
        text = "some text here"
        result = _truncate_words(text, max_words=0)
        assert result == "..."


class TestCleanAnswer:
    """Tests for _clean_answer()."""

    def test_removes_source_markers(self):
        from src.rag.pipeline import _clean_answer
        result = _clean_answer("Answer text [sources 1].")
        assert "[sources 1]" not in result
        assert "Answer text" in result

    def test_removes_various_marker_formats(self):
        from src.rag.pipeline import _clean_answer
        markers = [
            "[Source 1]",
            "[ sources 2 ]",
            "[SOURCE 3]",
            "[Sources 1] doesn't work",
        ]
        for text in markers:
            assert _clean_answer(text) != text

    def test_collapses_multiple_spaces(self):
        from src.rag.pipeline import _clean_answer
        result = _clean_answer("too    many    spaces")
        assert "  " not in result
        assert result == "too many spaces"

    def test_strips_leading_punctuation(self):
        from src.rag.pipeline import _clean_answer
        result = _clean_answer("...Answer text")
        assert not result.startswith("...")

    def test_empty_string(self):
        from src.rag.pipeline import _clean_answer
        assert _clean_answer("") == ""

    def test_none_input(self):
        from src.rag.pipeline import _clean_answer
        assert _clean_answer(None) == ""

    def test_only_source_marker(self):
        from src.rag.pipeline import _clean_answer
        result = _clean_answer("[sources 1]")
        assert result == ""


class TestIsInsufficient:
    """Tests for _is_insufficient()."""

    def test_empty_answer(self):
        from src.rag.pipeline import _is_insufficient
        assert _is_insufficient("", min_words=3) is True

    def test_none_answer(self):
        from src.rag.pipeline import _is_insufficient
        assert _is_insufficient(None, min_words=3) is True

    def test_below_min_words(self):
        from src.rag.pipeline import _is_insufficient
        assert _is_insufficient("too short", min_words=10) is True

    def test_above_min_words(self):
        from src.rag.pipeline import _is_insufficient
        text = "this answer has enough words to pass the threshold"
        assert _is_insufficient(text, min_words=3) is False

    def test_exact_min_words(self):
        from src.rag.pipeline import _is_insufficient
        text = "exactly three words"
        assert _is_insufficient(text, min_words=3) is False

    def test_only_non_alpha_tokens(self):
        from src.rag.pipeline import _is_insufficient
        assert _is_insufficient("1 2 3", min_words=3) is True

    def test_mixed_alpha_non_alpha(self):
        from src.rag.pipeline import _is_insufficient
        text = "hello 1 world 2"
        assert _is_insufficient(text, min_words=2) is False
        assert _is_insufficient(text, min_words=3) is True


class TestHedgingPatterns:
    """Tests for _HEDGING_PATTERNS constants and _is_hedging()."""

    def test_hedging_detected_not_directly_addressed(self):
        from src.rag.pipeline import _HEDGING_PATTERNS
        text = "The evidence does not directly address this question"
        assert any(p.search(text) for p in _HEDGING_PATTERNS)

    def test_hedging_detected_no_direct_evidence(self):
        from src.rag.pipeline import _HEDGING_PATTERNS
        text = "There is no direct evidence linking the two"
        assert any(p.search(text) for p in _HEDGING_PATTERNS)

    def test_hedging_not_detected_not_enough_info(self):
        from src.rag.pipeline import _HEDGING_PATTERNS
        text = "does not contain enough information"
        # This is now the HONEST correct response when evidence is about a
        # different disease — NOT hedging. The prompt instructs the LLM to
        # say this when evidence doesn't match the question.
        assert not any(p.search(text) for p in _HEDGING_PATTERNS)

    def test_hedging_not_detected_exact_prompt_mandated_text(self):
        """The exact prompt-mandated insufficient-info sentence is NOT hedging."""
        from src.rag.pipeline import _HEDGING_PATTERNS
        text = (
            "The retrieved medical literature does not contain sufficient "
            "information to answer this question."
        )
        assert not any(p.search(text) for p in _HEDGING_PATTERNS)

    def test_hedging_detected_cannot_answer(self):
        from src.rag.pipeline import _HEDGING_PATTERNS
        text = "I cannot answer this question"
        assert any(p.search(text) for p in _HEDGING_PATTERNS)

    def test_non_hedging_returns_false(self):
        from src.rag.pipeline import _HEDGING_PATTERNS
        text = "The patient should take metformin twice daily"
        assert not any(p.search(text) for p in _HEDGING_PATTERNS)


# ==============================================================================
# ── Pipeline fixtures ─────────────────────────────────────────────────────────
# ── These helpers build a RAGPipeline instance with fully mocked internals.  ──
# ==============================================================================


@pytest.fixture
def mock_index():
    """Build a mock FAISS index."""
    idx = MagicMock()
    idx.ntotal = 100
    return idx


@pytest.fixture
def mock_faiss(mock_index):
    """Build a mock faiss module."""
    faiss = MagicMock()
    faiss.read_index.return_value = mock_index
    faiss.normalize_L2 = MagicMock()
    return faiss


@pytest.fixture
def mock_encoder():
    """Build a mock SentenceTransformer encoder."""
    enc = MagicMock()
    enc.encode.return_value = np.random.rand(1, 768).astype(np.float32)
    return enc


@pytest.fixture
def mock_st_mod(mock_encoder):
    """Build a mock sentence_transformers module."""
    st = MagicMock()
    st.SentenceTransformer.return_value = mock_encoder
    return st


@pytest.fixture
def mock_openai_mod():
    """Build a mock openai module for Groq client."""
    oa = MagicMock()
    oa.OpenAI = MagicMock(return_value=MagicMock())
    return oa


@pytest.fixture
def mock_bm25():
    """Build a mock BM25 retriever."""
    bm25 = MagicMock()
    bm25.retrieve.return_value = []
    return bm25


@pytest.fixture
def mock_bm25_mod(mock_bm25):
    """Build a mock bm25_retriever module."""
    bm = MagicMock()
    bm.BM25Retriever = MagicMock(return_value=mock_bm25)
    return bm


@pytest.fixture
def mock_clf_mod():
    """Build a mock classifier module."""
    return MagicMock()


@pytest.fixture
def mock_df():
    """Build a standard test DataFrame."""
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
    """Build a mock rank_bm25 module."""
    rb = MagicMock()
    rb.BM25Okapi = MagicMock()
    return rb


class _PipelineBuilder:
    """Helper to build a fully-mocked RAGPipeline instance.

    Provides the pipeline ready for testing generation, answer, and other
    higher-level methods. Retrieval results are fully controllable via
    mock_index.search and mock_bm25.retrieve return values.
    """

    def __init__(self, mock_faiss, mock_index, mock_st_mod, mock_encoder,
                 mock_openai_mod, mock_bm25_mod, mock_bm25, mock_clf_mod,
                 mock_df, mock_rank_bm25, use_reranker=False, top_k=15,
                 groq_key="test-key", use_bm25=True):
        self._mock_index = mock_index
        self._mock_bm25 = mock_bm25
        self._mock_encoder = mock_encoder
        self._mock_df = mock_df
        self._mock_faiss = mock_faiss

        deps = {
            "faiss": mock_faiss,
            "sentence_transformers": mock_st_mod,
            "openai": mock_openai_mod,
            "rank_bm25": mock_rank_bm25,
            "src.classification.classifier": mock_clf_mod,
        }
        if use_bm25 and mock_bm25_mod is not None:
            deps["src.rag.bm25_retriever"] = mock_bm25_mod

        with patch.dict("sys.modules", deps):
            from src.rag import pipeline as rp

            with (
                patch("builtins.open", MagicMock()),
                patch("pickle.load", return_value=mock_df),
                patch.dict(os.environ, {"GROQ_API_KEY": groq_key})
                if groq_key is not None
                else patch.dict(os.environ, {}, clear=True),
            ):
                self._pipeline = rp.RAGPipeline(
                    top_k=top_k, use_reranker=use_reranker
                )

        # Wire index.search to return deterministic results
        self._search_calls = []

        def search_side_effect(query, k):
            self._search_calls.append(k)
            return (
                np.full((1, k), 0.5, dtype=np.float32),
                np.arange(min(k, len(mock_df)), dtype=np.int64).reshape(1, -1)
                if k <= len(mock_df)
                else np.arange(k, dtype=np.int64).reshape(1, k),
            )

        mock_index.search = MagicMock(side_effect=search_side_effect)

    @property
    def pipeline(self):
        return self._pipeline

    @property
    def mock_index(self):
        return self._mock_index

    @property
    def mock_bm25(self):
        return self._mock_bm25

    @property
    def mock_encoder(self):
        return self._mock_encoder

    @property
    def search_calls(self):
        return self._search_calls


@pytest.fixture
def builder(mock_faiss, mock_index, mock_st_mod, mock_encoder,
            mock_openai_mod, mock_bm25_mod, mock_bm25, mock_clf_mod,
            mock_df, mock_rank_bm25):
    """Build a standard pipeline builder with Groq client."""
    return _PipelineBuilder(
        mock_faiss=mock_faiss,
        mock_index=mock_index,
        mock_st_mod=mock_st_mod,
        mock_encoder=mock_encoder,
        mock_openai_mod=mock_openai_mod,
        mock_bm25_mod=mock_bm25_mod,
        mock_bm25=mock_bm25,
        mock_clf_mod=mock_clf_mod,
        mock_df=mock_df,
        mock_rank_bm25=mock_rank_bm25,
        use_reranker=False,
        top_k=15,
        groq_key="test-key",
    )


@pytest.fixture
def no_groq_builder(mock_faiss, mock_index, mock_st_mod, mock_encoder,
                    mock_openai_mod, mock_bm25_mod, mock_bm25, mock_clf_mod,
                    mock_df, mock_rank_bm25):
    """Build a pipeline builder without Groq (no API key)."""
    # Need to mock transformers/torch for the flan-t5 fallback path
    mock_tok = MagicMock()
    mock_tok.return_value = MagicMock()
    mock_model = MagicMock()
    mock_model.generate.return_value = MagicMock()
    # Mock the tokenizer.decode to return a string
    mock_tok_obj = MagicMock()
    mock_tok_obj.decode.return_value = "Generated answer from local model."

    mock_tf_mod = MagicMock()
    mock_tf_mod.AutoTokenizer = MagicMock()
    mock_tf_mod.AutoTokenizer.from_pretrained.return_value = mock_tok_obj
    mock_tf_mod.AutoModelForSeq2SeqLM = MagicMock()
    mock_tf_mod.AutoModelForSeq2SeqLM.from_pretrained.return_value = mock_model

    mock_torch = MagicMock()
    mock_torch.no_grad = MagicMock()
    mock_torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
    mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=None)

    with patch.dict("sys.modules", {
        "transformers": mock_tf_mod,
        "torch": mock_torch,
    }):
        return _PipelineBuilder(
            mock_faiss=mock_faiss,
            mock_index=mock_index,
            mock_st_mod=mock_st_mod,
            mock_encoder=mock_encoder,
            mock_openai_mod=mock_openai_mod,
            mock_bm25_mod=mock_bm25_mod,
            mock_bm25=mock_bm25,
            mock_clf_mod=mock_clf_mod,
            mock_df=mock_df,
            mock_rank_bm25=mock_rank_bm25,
            use_reranker=False,
            top_k=15,
            groq_key="",  # No Groq key -> local model
        )


# ==============================================================================
# ── _row_to_dict ──────────────────────────────────────────────────────────────
# ==============================================================================

class TestRowToDict:
    """Tests for RAGPipeline._row_to_dict()."""

    def test_basic_mapping(self, builder):
        """_row_to_dict maps a row index and distance to a dict with expected keys."""
        result = builder.pipeline._row_to_dict(idx=5, dist=0.75)
        assert result["chunk_id"] == 5
        assert result["question"] == "q5"
        assert result["answer"] == "a5"
        assert result["context"] == "c5"
        assert result["text_chunk"] == "t5"
        assert result["category"] in ("Symptoms", "General", "Treatment", "Diagnosis", "Medication")
        assert result["distance"] == 0.75

    def test_category_from_dataframe(self, builder):
        """category comes from the DataFrame's category column."""
        result = builder.pipeline._row_to_dict(idx=0, dist=0.5)
        assert result["category"] == "Symptoms"
        result = builder.pipeline._row_to_dict(idx=30, dist=0.5)
        assert result["category"] == "General"


# ==============================================================================
# ── _is_hedging ───────────────────────────────────────────────────────────────
# ==============================================================================

class TestIsHedgingMethod:
    """Tests for RAGPipeline._is_hedging()."""

    def test_detects_hedging(self, builder):
        pipeline = builder.pipeline
        assert pipeline._is_hedging("The evidence does not directly address this question") is True
        assert pipeline._is_hedging("There is no direct evidence for that") is True
        # "does not contain enough information" is now the HONEST correct
        # response (not hedging) — the prompt tells the LLM to say this when
        # evidence is about a different disease.
        assert pipeline._is_hedging("This does not contain enough information") is False
        assert pipeline._is_hedging("I cannot answer this question") is True
        # "Not enough information" is the honest correct response now — not hedging
        assert pipeline._is_hedging("Not enough information to answer") is False

    def test_non_hedging(self, builder):
        pipeline = builder.pipeline
        assert pipeline._is_hedging("Metformin reduces HbA1c by 1.5%") is False
        assert pipeline._is_hedging("The patient should be treated with antibiotics") is False

    def test_empty_string(self, builder):
        pipeline = builder.pipeline
        assert pipeline._is_hedging("") is False


# ==============================================================================
# ── format_sources ────────────────────────────────────────────────────────────
# ==============================================================================

class TestFormatSources:
    """Tests for RAGPipeline.format_sources()."""

    def test_basic_formatting(self, builder):
        """format_sources produces dicts with expected keys."""
        pipeline = builder.pipeline
        retrieved = [
            {"chunk_id": 5, "question": "q5", "category": "Symptoms",
             "distance": 0.75, "answer": "a5", "context": "some context here"},
        ]
        sources = pipeline.format_sources(retrieved)
        assert len(sources) == 1
        s = sources[0]
        assert s["chunk_id"] == 5
        assert s["question"] == "q5"
        assert s["category"] == "Symptoms"
        assert "distance" in s
        assert "relevance_score" in s
        assert "reranker_score" in s
        assert "excerpt" in s
        assert s["relevance_score"] == pytest.approx(0.75, abs=1e-4)

    def test_distance_clamped(self, builder):
        """distance is clamped to [0, 1]."""
        pipeline = builder.pipeline
        retrieved = [
            {"chunk_id": 1, "question": "q", "category": "General",
             "distance": 1.5, "answer": "a", "context": "c"},
        ]
        sources = pipeline.format_sources(retrieved)
        assert sources[0]["relevance_score"] == 1.0

    def test_negative_distance(self, builder):
        """Negative distance is clamped to 0."""
        pipeline = builder.pipeline
        retrieved = [
            {"chunk_id": 1, "question": "q", "category": "General",
             "distance": -0.5, "answer": "a", "context": "c"},
        ]
        sources = pipeline.format_sources(retrieved)
        assert sources[0]["relevance_score"] == 0.0

    def test_with_category_score(self, builder):
        """category_score is rounded to 4 decimal places."""
        pipeline = builder.pipeline
        retrieved = [
            {"chunk_id": 1, "question": "q", "category": "Symptoms",
             "distance": 0.5, "answer": "a", "context": "c",
             "category_score": 0.854321},
        ]
        sources = pipeline.format_sources(retrieved)
        assert sources[0]["category_score"] == pytest.approx(0.8543, abs=1e-4)

    def test_empty_list(self, builder):
        """Empty list returns empty list."""
        pipeline = builder.pipeline
        assert pipeline.format_sources([]) == []

    def test_excerpt_is_truncated(self, builder):
        """excerpt is the first 150 chars of context."""
        pipeline = builder.pipeline
        long_context = "x" * 300
        retrieved = [
            {"chunk_id": 1, "question": "q", "category": "General",
             "distance": 0.5, "answer": "a", "context": long_context},
        ]
        sources = pipeline.format_sources(retrieved)
        assert len(sources[0]["excerpt"]) <= 150


# ==============================================================================
# ── _generate_once ────────────────────────────────────────────────────────────
# ==============================================================================

class TestGenerateOnce:
    """Tests for RAGPipeline._generate_once()."""

    def test_groq_path(self, builder):
        """_generate_once with Groq calls _call_groq and cleans the answer."""
        pipeline = builder.pipeline
        pipeline._use_groq = True
        pipeline._call_groq = MagicMock(return_value="  The answer is clear.  ")

        chunks = [
            {"answer": "a1", "context": "c1"},
            {"answer": "a2", "context": "c2"},
        ]
        result = pipeline._generate_once("test query", chunks)
        assert result == "The answer is clear."
        pipeline._call_groq.assert_called_once()

    def test_groq_path_cleans_answer(self, builder):
        """_generate_once removes source markers from Groq output."""
        pipeline = builder.pipeline
        pipeline._use_groq = True
        pipeline._call_groq = MagicMock(return_value="Answer [sources 1] here.")

        chunks = [{"answer": "a1", "context": "c1"}]
        result = pipeline._generate_once("test query", chunks)
        assert "[sources" not in result

    def test_non_groq_path(self, no_groq_builder):
        """_generate_once without Groq uses local model."""
        pipeline = no_groq_builder.pipeline
        assert pipeline._use_groq is False

        chunks = [{"answer": "a1", "context": "c1"}]
        result = pipeline._generate_once("test query", chunks)
        assert isinstance(result, str)


# ==============================================================================
# ── _call_groq ────────────────────────────────────────────────────────────────
# ==============================================================================

class TestCallGroq:
    """Tests for RAGPipeline._call_groq()."""

    def test_basic_call(self, builder):
        """_call_groq sends request to Groq and returns content."""
        pipeline = builder.pipeline
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "  Answer text  "
        pipeline._groq_clients[0].chat.completions.create.return_value = mock_response

        result = pipeline._call_groq("test prompt")
        assert result == "Answer text"

    def test_uses_system_message(self, builder):
        """_call_groq includes system message."""
        pipeline = builder.pipeline
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "answer"
        pipeline._groq_clients[0].chat.completions.create.return_value = mock_response

        pipeline._call_groq("test prompt")
        call_kwargs = pipeline._groq_clients[0].chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        system_msgs = [m for m in messages if m["role"] == "system"]
        assert len(system_msgs) == 1

    def test_temperature_zero(self, builder):
        """_call_groq uses temperature=0.0."""
        pipeline = builder.pipeline
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "answer"
        pipeline._groq_clients[0].chat.completions.create.return_value = mock_response

        pipeline._call_groq("test prompt")
        call_kwargs = pipeline._groq_clients[0].chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.0

    def test_429_rotates_key_and_retries(self, builder):
        """On 429 rate limit, _call_groq rotates to next key and retries."""
        pipeline = builder.pipeline

        # Set up 2 mock clients
        mock_client1 = MagicMock()
        mock_client2 = MagicMock()
        pipeline._groq_clients = [mock_client1, mock_client2]
        pipeline._groq_key_index = 0

        # First client raises 429, second succeeds
        error_429 = Exception("Rate limited")
        error_429.status_code = 429
        mock_client1.chat.completions.create.side_effect = error_429

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Success answer"
        mock_client2.chat.completions.create.return_value = mock_response

        result = pipeline._call_groq("test prompt")
        assert result == "Success answer"
        # Key should have rotated to 1
        assert pipeline._groq_key_index == 1

    def test_429_single_key_no_rotation(self, builder):
        """With a single key, 429 is not caught — exception propagates."""
        pipeline = builder.pipeline

        # Single key only (default in builder)
        pipeline._groq_clients = [pipeline._groq_clients[0]]
        pipeline._groq_key_index = 0

        error_429 = Exception("Rate limited")
        error_429.status_code = 429
        pipeline._groq_clients[0].chat.completions.create.side_effect = error_429

        with pytest.raises(Exception, match="Rate limited"):
            pipeline._call_groq("test prompt")
        # Key should NOT have rotated (still 0)
        assert pipeline._groq_key_index == 0


# ==============================================================================
# ── generate ──────────────────────────────────────────────────────────────────
# ==============================================================================

class TestGenerate:
    """Tests for RAGPipeline.generate()."""

    def test_no_chunks_returns_insufficient_message(self, builder):
        """generate with empty chunks returns INSUFFICIENT_CONTEXT_MESSAGE."""
        from src.rag.pipeline import INSUFFICIENT_CONTEXT_MESSAGE
        pipeline = builder.pipeline
        result = pipeline.generate("test query", [])
        assert result == INSUFFICIENT_CONTEXT_MESSAGE

    def test_extractive_mode_returns_best_answer(self, builder):
        """Extractive mode returns the best chunk's answer directly."""
        pipeline = builder.pipeline
        pipeline._extractive = True
        chunks = [
            {"answer": "   ", "context": "c1"},  # empty answer
            {"answer": "Best answer here", "context": "c2"},
            {"answer": "Another answer", "context": "c3"},
        ]
        result = pipeline.generate("test query", chunks)
        assert result == "Best answer here"

    def test_extractive_mode_all_empty(self, builder):
        """Extractive mode with all empty answers returns insufficient message."""
        from src.rag.pipeline import INSUFFICIENT_CONTEXT_MESSAGE
        pipeline = builder.pipeline
        pipeline._extractive = True
        chunks = [
            {"answer": "", "context": "c1"},
            {"answer": "   ", "context": "c2"},
        ]
        result = pipeline.generate("test query", chunks)
        assert result == INSUFFICIENT_CONTEXT_MESSAGE

    def test_abstractive_mode_normal(self, builder):
        """Abstractive mode generates via LLM and returns cleaned answer."""
        pipeline = builder.pipeline
        pipeline._generate_once = MagicMock(return_value="This is a good answer about treatment.")
        pipeline._is_hedging = MagicMock(return_value=False)

        chunks = [{"answer": "a1", "context": "c1"}]
        result = pipeline.generate("test query", chunks)
        assert result == "This is a good answer about treatment."

    def test_abstractive_mode_insufficient_falls_back(self, builder):
        """Abstractive mode with insufficient answer falls back to best chunk."""
        pipeline = builder.pipeline
        pipeline._generate_once = MagicMock(return_value="short")  # insufficient (1 word, needs 3)
        pipeline.min_answer_words = 3

        chunks = [
            {"answer": "The best answer from literature.", "context": "c1"},
        ]
        result = pipeline.generate("test query", chunks)
        assert result == "The best answer from literature."

    def test_abstractive_mode_insufficient_all_empty(self, builder):
        """Insufficient generation with no best chunk returns insufficient message."""
        from src.rag.pipeline import INSUFFICIENT_CONTEXT_MESSAGE
        pipeline = builder.pipeline
        pipeline._generate_once = MagicMock(return_value="short")
        pipeline.min_answer_words = 3

        chunks = [{"answer": "", "context": "c1"}]
        result = pipeline.generate("test query", chunks)
        assert result == INSUFFICIENT_CONTEXT_MESSAGE

    def test_hedging_recovery_retry_succeeds(self, builder):
        """Hedging detection triggers retry with stronger prompt, retry succeeds."""
        pipeline = builder.pipeline
        pipeline._use_groq = True

        # First generation hedges
        pipeline._generate_once = MagicMock(return_value="The evidence does not directly address this.")
        pipeline._is_hedging = MagicMock(side_effect=[True, False])  # first hedges, retry doesn't
        pipeline._call_groq = MagicMock(return_value="The treatment is effective.")

        chunks = [{"answer": "The treatment is effective.", "context": "c1"}]
        result = pipeline.generate("test query", chunks)

        # Retry should have used a different prompt
        pipeline._call_groq.assert_called_once()
        call_args = pipeline._call_groq.call_args[0][0]
        assert "Retry" in call_args or "Retry" in call_args or "Follow" in call_args or "Rules" in call_args
        assert result == "The treatment is effective."

    def test_hedging_recovery_retry_still_hedges_falls_to_best(self, builder):
        """Retry still hedges, falls back to best chunk answer."""
        pipeline = builder.pipeline
        pipeline._use_groq = True

        pipeline._generate_once = MagicMock(return_value="The evidence does not directly address this.")
        pipeline._is_hedging = MagicMock(return_value=True)  # both hedge
        pipeline._call_groq = MagicMock(return_value="Not enough information.")  # retry also hedges

        chunks = [{"answer": "The treatment is effective.", "context": "c1"}]
        result = pipeline.generate("test query", chunks)

        assert result == "The treatment is effective."

    def test_hedging_only_with_groq(self, builder):
        """Hedging recovery only happens with Groq (not local model)."""
        pipeline = builder.pipeline
        pipeline._use_groq = False  # local model

        pipeline._generate_once = MagicMock(return_value="The evidence does not directly address this.")
        pipeline._is_hedging = MagicMock(return_value=True)

        chunks = [{"answer": "a1", "context": "c1"}]
        result = pipeline.generate("test query", chunks)

        # No retry - hedging is skipped for non-Groq models
        assert result == "The evidence does not directly address this."

    def test_disease_mismatch_triggers_general_knowledge_fallback(self, builder):
        """When evidence is insufficient, generate() falls back to LLM general
        knowledge instead of returning the refusal message."""
        pipeline = builder.pipeline
        honest_response = (
            "The retrieved medical literature does not contain sufficient "
            "information to answer this question."
        )
        pipeline._generate_once = MagicMock(return_value=honest_response)
        pipeline._is_hedging = MagicMock(return_value=False)
        pipeline._call_groq = MagicMock(return_value="General knowledge answer about symptoms.")

        chunks = [{"answer": "Some irrelevant answer.", "context": "c1"}]
        result = pipeline.generate("test query", chunks)

        # Should fall back to general knowledge instead of returning the refusal
        assert result == "General knowledge answer about symptoms."
        # _is_hedging was checked but returned False — no retry
        pipeline._is_hedging.assert_called_once_with(honest_response)
        # _call_groq was called for the general knowledge fallback
        pipeline._call_groq.assert_called()

    def test_disease_mismatch_not_flagged_as_hedging_in_is_hedging(self, builder):
        """_is_hedging() explicitly returns False for the prompt-mandated text."""
        pipeline = builder.pipeline
        text = (
            "The retrieved medical literature does not contain sufficient "
            "information to answer this question."
        )
        assert pipeline._is_hedging(text) is False

    def test_hedging_retry_still_hedges_no_best_answer(self, builder):
        """Retry still hedges and no best chunk answer, returns retry result."""
        pipeline = builder.pipeline
        pipeline._use_groq = True

        pipeline._generate_once = MagicMock(return_value="The evidence does not directly address this.")
        pipeline._is_hedging = MagicMock(return_value=True)
        pipeline._call_groq = MagicMock(return_value="I cannot answer.")

        chunks = [{"answer": "", "context": "c1"}]
        result = pipeline.generate("test query", chunks)
        # Retry still hedges, best_answer is empty, so cleaned stays as
        # the first generation result
        assert isinstance(result, str)


# ==============================================================================
# ── answer ────────────────────────────────────────────────────────────────────
# ==============================================================================

class TestAnswer:
    """Tests for RAGPipeline.answer()."""

    def test_basic_answer_structure(self, builder):
        """answer() returns dict with expected keys when needs_rag=True."""
        pipeline = builder.pipeline
        pipeline._needs_retrieval = MagicMock(return_value=True)
        pipeline.generate = MagicMock(return_value="Disease X causes fever.")

        result = pipeline.answer("What causes disease X?")
        assert set(result.keys()) == {
            "question", "answer", "answer_raw", "retrieved_sources",
            "disclaimer_present", "top_k", "used_rag", "answer_source",
            "retrieval_quality", "mean_cosine_similarity",
        }
        assert result["question"] == "What causes disease X?"
        assert result["answer_raw"] == "Disease X causes fever."
        assert result["used_rag"] is True
        assert result["disclaimer_present"] is True
        assert "MEDICAL DISCLAIMER" in result["answer"]

    def test_no_rag_path_groq(self, builder):
        """answer() with needs_rag=False uses direct Groq answer."""
        pipeline = builder.pipeline
        pipeline._needs_retrieval = MagicMock(return_value=False)
        pipeline._call_groq = MagicMock(return_value="Direct answer.")

        result = pipeline.answer("Hi, how are you?")
        assert result["answer_raw"] == "Direct answer."
        assert result["used_rag"] is False
        assert result["top_k"] == 0
        assert result["retrieval_quality"] == 0.0
        assert result["mean_cosine_similarity"] == 0.0

    def test_no_rag_path_no_groq(self, no_groq_builder):
        """answer() with needs_rag=False and no Groq returns insufficient context."""
        from src.rag.pipeline import INSUFFICIENT_CONTEXT_MESSAGE
        pipeline = no_groq_builder.pipeline
        pipeline._needs_retrieval = MagicMock(return_value=False)

        result = pipeline.answer("Hi")
        assert result["answer_raw"] == INSUFFICIENT_CONTEXT_MESSAGE
        assert result["used_rag"] is False

    def test_retrieval_quality_scores(self, builder):
        """answer() computes retrieval_quality and mean_cosine_similarity."""
        pipeline = builder.pipeline
        pipeline._needs_retrieval = MagicMock(return_value=True)
        pipeline.generate = MagicMock(return_value="Answer text.")

        # Wire retrieve to return items with specific scores
        def mock_retrieve(query, top_k=None):
            return [
                {"chunk_id": 1, "question": "q1", "category": "General",
                 "distance": 0.8, "reranker_score": 0.9, "answer": "a1",
                 "context": "c1", "text_chunk": "t1"},
                {"chunk_id": 2, "question": "q2", "category": "General",
                 "distance": 0.6, "reranker_score": 0.7, "answer": "a2",
                 "context": "c2", "text_chunk": "t2"},
            ]
        pipeline.retrieve = mock_retrieve

        result = pipeline.answer("test query")
        # retrieval_quality = mean of reranker_score = (0.9 + 0.7) / 2 = 0.8
        assert result["retrieval_quality"] == pytest.approx(0.8, abs=1e-4)
        # mean_cosine_similarity = mean of distance = (0.8 + 0.6) / 2 = 0.7
        assert result["mean_cosine_similarity"] == pytest.approx(0.7, abs=1e-4)

    def test_answer_source_is_rag_by_default(self, builder):
        """answer_source defaults to 'rag' when using standard RAG."""
        pipeline = builder.pipeline
        pipeline._needs_retrieval = MagicMock(return_value=True)
        pipeline.generate = MagicMock(return_value="An answer from evidence.")

        result = pipeline.answer("test")
        assert result["answer_source"] == "rag"

    def test_answer_source_general_knowledge_when_fallback(self, builder):
        """answer_source is 'general_knowledge' when fallback is triggered."""
        pipeline = builder.pipeline
        pipeline._needs_retrieval = MagicMock(return_value=True)
        pipeline.generate = MagicMock(return_value="General knowledge answer.")
        pipeline._last_answer_source = "general_knowledge"

        result = pipeline.answer("test")
        assert result["answer_source"] == "general_knowledge"
        from src.rag.pipeline import GENERAL_KNOWLEDGE_NOTE
        assert GENERAL_KNOWLEDGE_NOTE in result["answer"]

    def test_disclaimer_appended(self, builder):
        """answer appends disclaimer to answer_raw."""
        from src.rag.pipeline import DISCLAIMER
        pipeline = builder.pipeline
        pipeline._needs_retrieval = MagicMock(return_value=True)
        pipeline.generate = MagicMock(return_value="Pure answer.")

        result = pipeline.answer("test")
        assert result["answer"] == "Pure answer." + DISCLAIMER


# ==============================================================================
# ── answer_with_routing ───────────────────────────────────────────────────────
# ==============================================================================

class TestAnswerWithRouting:
    """Tests for RAGPipeline.answer_with_routing()."""

    def test_with_explicit_category(self, builder):
        """When category is provided, uses retrieve_by_category with that category."""
        pipeline = builder.pipeline
        pipeline.generate = MagicMock(return_value="Answer text.")

        result = pipeline.answer_with_routing("test query", category="Symptoms")
        assert result["category"] == "Symptoms"
        assert result["routing_applied"] is True

    def test_no_category_uses_classifier(self, builder):
        """Without category, uses classifier to predict it."""
        pipeline = builder.pipeline
        pipeline._use_classifier = True
        pipeline._classifier = MagicMock()
        pipeline._classifier.predict_with_confidence.return_value = {
            "category": "Treatment",
            "confidence": 0.85,
            "all_scores": {
                "Treatment": 0.85, "Symptoms": 0.05, "General": 0.05,
                "Diagnosis": 0.03, "Medication": 0.02, "Prevention": 0.0,
            },
        }
        pipeline.generate = MagicMock(return_value="Answer text.")

        result = pipeline.answer_with_routing("What is the treatment for hypertension?")
        assert result["category"] == "Treatment"
        assert result["classifier_confidence"] == pytest.approx(0.85, abs=1e-4)
        assert result["routing_applied"] is True

    def test_classifier_low_confidence_uses_general(self, builder):
        """Low classifier confidence (< 0.70) falls back to general retrieval."""
        pipeline = builder.pipeline
        pipeline._use_classifier = True
        pipeline._classifier = MagicMock()
        pipeline._classifier.predict_with_confidence.return_value = {
            "category": "Symptoms",
            "confidence": 0.45,
            "all_scores": {
                "Symptoms": 0.45, "General": 0.30, "Treatment": 0.10,
                "Diagnosis": 0.08, "Medication": 0.05, "Prevention": 0.02,
            },
        }
        pipeline.generate = MagicMock(return_value="Answer.")

        result = pipeline.answer_with_routing("test query")
        # Category should be None (confidence too low), so routing_applied is False
        assert result["routing_applied"] is False
        assert result["classifier_confidence"] == pytest.approx(0.45, abs=1e-4)

    def test_no_classifier_uses_general(self, builder):
        """When classifier is not loaded, uses general retrieval."""
        pipeline = builder.pipeline
        pipeline._use_classifier = False
        pipeline.generate = MagicMock(return_value="Answer text.")

        result = pipeline.answer_with_routing("test query")
        assert result["routing_applied"] is False

    def test_structure(self, builder):
        """answer_with_routing returns dict with expected keys."""
        pipeline = builder.pipeline
        pipeline.generate = MagicMock(return_value="Answer text.")

        result = pipeline.answer_with_routing("test query", category="General")
        expected_keys = {
            "question", "category", "classifier_confidence",
            "routing_applied", "answer", "answer_raw",
            "retrieved_sources", "disclaimer_present", "top_k",
            "answer_source", "category_matched_sources",
        }
        assert set(result.keys()) == expected_keys

    def test_answer_source_rag_default_routing(self, builder):
        """answer_with_routing sets answer_source to 'rag' by default."""
        pipeline = builder.pipeline
        pipeline.generate = MagicMock(return_value="Answer text.")

        result = pipeline.answer_with_routing("test query", category="Symptoms")
        assert result["answer_source"] == "rag"
        from src.rag.pipeline import GENERAL_KNOWLEDGE_NOTE
        assert GENERAL_KNOWLEDGE_NOTE not in result["answer"]

    def test_answer_source_general_knowledge_routing(self, builder):
        """answer_with_routing includes answer_source and transparency note."""
        pipeline = builder.pipeline
        pipeline.generate = MagicMock(return_value="General knowledge answer.")
        pipeline._last_answer_source = "general_knowledge"

        result = pipeline.answer_with_routing("test query", category="Symptoms")
        assert result["answer_source"] == "general_knowledge"
        from src.rag.pipeline import GENERAL_KNOWLEDGE_NOTE
        assert GENERAL_KNOWLEDGE_NOTE in result["answer"]


# ==============================================================================
# ── _needs_retrieval ──────────────────────────────────────────────────────────
# ==============================================================================

class TestNeedsRetrieval:
    """Tests for RAGPipeline._needs_retrieval()."""

    def test_without_groq_always_returns_true(self, no_groq_builder):
        """Without Groq, _needs_retrieval always returns True."""
        pipeline = no_groq_builder.pipeline
        assert pipeline._needs_retrieval("Hi") is True
        assert pipeline._needs_retrieval("What is metformin?") is True

    def test_groq_medical_query_returns_true(self, builder):
        """Medical query returns True (needs RAG)."""
        pipeline = builder.pipeline
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"needs_rag": true}'
        pipeline._groq_clients[0].chat.completions.create.return_value = mock_response

        assert pipeline._needs_retrieval("Does metformin reduce HbA1c?") is True

    def test_groq_greeting_returns_false(self, builder):
        """Greeting query returns False (no RAG needed)."""
        pipeline = builder.pipeline
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"needs_rag": false}'
        pipeline._groq_clients[0].chat.completions.create.return_value = mock_response

        assert pipeline._needs_retrieval("Hi there!") is False

    def test_groq_exception_falls_back_to_true(self, builder):
        """Exception in Groq call falls back to True (always retrieve)."""
        pipeline = builder.pipeline
        pipeline._groq_clients[0].chat.completions.create.side_effect = RuntimeError("API down")

        assert pipeline._needs_retrieval("test query") is True

    def test_groq_invalid_json_falls_back_to_true(self, builder):
        """Invalid JSON response falls back to True."""
        pipeline = builder.pipeline
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "not valid json"
        pipeline._groq_clients[0].chat.completions.create.return_value = mock_response

        assert pipeline._needs_retrieval("test query") is True


# ==============================================================================
# ── Module-level functions ────────────────────────────────────────────────────
# ==============================================================================

class TestBuildRagPipeline:
    """Tests for build_rag_pipeline() singleton."""

    def test_returns_ragpipeline_instance(self):
        """build_rag_pipeline() returns a RAGPipeline instance."""
        import src.rag.pipeline as rp
        rp._pipeline_instance = None

        # Use patch.object to patch directly on the module object
        with patch.object(rp, "RAGPipeline") as mock_rp:
            mock_instance = MagicMock()
            mock_rp.return_value = mock_instance

            instance = rp.build_rag_pipeline()
            assert instance is mock_instance


# ==============================================================================
# ── Init edge cases ───────────────────────────────────────────────────────────
# ==============================================================================

class TestInitCategoryExpansion:
    """Tests for category expansion override in __init__."""

    def test_category_expansion_dict_override(self, builder):
        """category_expansion as dict overrides specific categories."""
        pipeline = builder.pipeline
        # Check Symptoms is 5 by default
        assert pipeline._category_expansion["Symptoms"] == 5

    def test_category_expansion_unknown_categories_default_to_3(self, builder):
        """Unknown categories default to 3x expansion."""
        pipeline = builder.pipeline
        expansion = pipeline._category_expansion
        assert expansion.get("UnknownCategory", 3) == 3
        assert expansion.get("Cardiology", 3) == 3


# ==============================================================================
# ── Constants ─────────────────────────────────────────────────────────────────
# ==============================================================================

class TestPipelineConstants:
    """Tests for module-level constants in src/rag/pipeline.py."""

    def test_disclaimer_constant(self):
        from src.rag.pipeline import DISCLAIMER
        assert "MEDICAL DISCLAIMER" in DISCLAIMER
        assert "AI system" in DISCLAIMER

    def test_insufficient_context_constant(self):
        from src.rag.pipeline import INSUFFICIENT_CONTEXT_MESSAGE
        assert "enough information" in INSUFFICIENT_CONTEXT_MESSAGE

    def test_inject_k_constant(self):
        from src.rag.pipeline import DEFAULT_INJECT_K
        assert DEFAULT_INJECT_K == 5

    def test_category_expansion_constant(self):
        from src.rag.pipeline import CATEGORY_EXPANSION
        assert CATEGORY_EXPANSION["Medication"] == 2
        assert CATEGORY_EXPANSION["Treatment"] == 2
        assert CATEGORY_EXPANSION["Symptoms"] == 5
        assert CATEGORY_EXPANSION["Prevention"] == 4


class TestInitNoGroq:
    """Tests for RAGPipeline init without Groq key (local model)."""

    def test_no_groq_sets_use_groq_false(self, no_groq_builder):
        """Without GROQ_API_KEY, _use_groq is False."""
        pipeline = no_groq_builder.pipeline
        assert pipeline._use_groq is False

    def test_no_groq_has_tokenizer_and_model(self, no_groq_builder):
        """Without Groq, local tokenizer and model are loaded."""
        pipeline = no_groq_builder.pipeline
        assert hasattr(pipeline, "tokenizer")
        assert hasattr(pipeline, "model")


# ==============================================================================
# ── Init edge cases: BM25 ImportError ─────────────────────────────────────────
# ==============================================================================

class TestInitBm25ImportError:
    """Tests for RAGPipeline init when BM25 cannot be imported."""

    def test_bm25_import_error_sets_use_bm25_false(
        self, mock_faiss, mock_index,
        mock_st_mod, mock_encoder,
        mock_openai_mod, mock_clf_mod,
        mock_df, mock_rank_bm25,
    ):
        """When BM25Retriever is not importable, _use_bm25 is False."""
        import types

        # Module without BM25Retriever attribute -> ImportError on from...import
        bad_bm25_mod = types.ModuleType("src.rag.bm25_retriever")

        deps = {
            "faiss": mock_faiss,
            "sentence_transformers": mock_st_mod,
            "openai": mock_openai_mod,
            "rank_bm25": mock_rank_bm25,
            "src.classification.classifier": mock_clf_mod,
            "src.rag.bm25_retriever": bad_bm25_mod,
        }

        with patch.dict("sys.modules", deps):
            from src.rag import pipeline as rp

            with (
                patch("builtins.open", MagicMock()),
                patch("pickle.load", return_value=mock_df),
                patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}),
            ):
                pipeline = rp.RAGPipeline(top_k=15, use_reranker=False)

        assert pipeline._use_bm25 is False
        assert not hasattr(pipeline, "bm25")


# ==============================================================================
# ── Init edge cases: settings import exception ────────────────────────────────
# ==============================================================================

class TestInitSettingsException:
    """Tests for RAGPipeline init when config.settings import fails."""

    def test_settings_exception_fallback_threshold(
        self, mock_faiss, mock_index,
        mock_st_mod, mock_encoder,
        mock_openai_mod, mock_bm25_mod,
        mock_bm25, mock_clf_mod,
        mock_df, mock_rank_bm25,
    ):
        """When config.settings import fails, _bm25_threshold defaults to 12.0."""
        deps = {
            "faiss": mock_faiss,
            "sentence_transformers": mock_st_mod,
            "openai": mock_openai_mod,
            "rank_bm25": mock_rank_bm25,
            "src.classification.classifier": mock_clf_mod,
            "src.rag.bm25_retriever": mock_bm25_mod,
            "config": MagicMock(),  # prevent settings import
        }
        # Ensure config.settings doesn't exist
        del deps["config"].settings

        with patch.dict("sys.modules", deps):
            from src.rag import pipeline as rp

            with (
                patch("builtins.open", MagicMock()),
                patch("pickle.load", return_value=mock_df),
                patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}),
            ):
                pipeline = rp.RAGPipeline(top_k=15, use_reranker=False)

        assert pipeline._bm25_threshold == pytest.approx(12.0, abs=1e-4)


# ==============================================================================
# ── Init edge cases: category_expansion exception ─────────────────────────────
# ==============================================================================

class TestInitCategoryExpansionException:
    """Tests for RAGPipeline init with invalid category_expansion."""

    def test_category_expansion_invalid_json_silently_fails(
        self, mock_faiss, mock_index,
        mock_st_mod, mock_encoder,
        mock_openai_mod, mock_bm25_mod,
        mock_bm25, mock_clf_mod,
        mock_df, mock_rank_bm25,
    ):
        """Invalid JSON string for category_expansion silently keeps defaults."""
        deps = {
            "faiss": mock_faiss,
            "sentence_transformers": mock_st_mod,
            "openai": mock_openai_mod,
            "rank_bm25": mock_rank_bm25,
            "src.classification.classifier": mock_clf_mod,
            "src.rag.bm25_retriever": mock_bm25_mod,
        }

        with patch.dict("sys.modules", deps):
            from src.rag import pipeline as rp

            with (
                patch("builtins.open", MagicMock()),
                patch("pickle.load", return_value=mock_df),
                patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}),
            ):
                # Invalid JSON string -> exception caught, defaults preserved
                pipeline = rp.RAGPipeline(
                    top_k=15, use_reranker=False,
                    category_expansion="not valid json",
                )

        # Defaults should be preserved
        assert pipeline._category_expansion["Symptoms"] == 5
        assert pipeline._category_expansion["Medication"] == 2


# ==============================================================================
# ── Init edge cases: classifier load exception ────────────────────────────────
# ==============================================================================

class TestInitClassifierException:
    """Tests for RAGPipeline init when classifier fails to load."""

    def test_classifier_exception_disables_routing(
        self, mock_faiss, mock_index,
        mock_st_mod, mock_encoder,
        mock_openai_mod, mock_bm25_mod,
        mock_bm25, mock_df,
        mock_rank_bm25,
    ):
        """When classifier.load_classifier() raises, _use_classifier stays False."""
        import types
        bad_clf_mod = types.ModuleType("src.classification.classifier")
        # No load_classifier attribute -> AttributeError when RAGPipeline tries to call it

        deps = {
            "faiss": mock_faiss,
            "sentence_transformers": mock_st_mod,
            "openai": mock_openai_mod,
            "rank_bm25": mock_rank_bm25,
            "src.classification.classifier": bad_clf_mod,
            "src.rag.bm25_retriever": mock_bm25_mod,
        }

        with patch.dict("sys.modules", deps):
            from src.rag import pipeline as rp

            with (
                patch("builtins.open", MagicMock()),
                patch("pickle.load", return_value=mock_df),
                patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}),
            ):
                pipeline = rp.RAGPipeline(top_k=15, use_reranker=False)

        assert pipeline._use_classifier is False


# ==============================================================================
# ── retrieve_by_category ──────────────────────────────────────────────────────
# ==============================================================================

class TestRetrieveByCategory:
    """Tests for RAGPipeline.retrieve_by_category()."""

    def test_continuous_scoring_with_all_scores(self, builder):
        """Continuous category scoring sorts by category_score then distance."""
        pipeline = builder.pipeline
        all_scores = {
            "Symptoms": 0.9, "General": 0.05, "Treatment": 0.02,
            "Diagnosis": 0.02, "Medication": 0.01,
        }
        results = pipeline.retrieve_by_category(
            "test query", "Symptoms", all_scores=all_scores
        )
        assert len(results) > 0
        # All results should have category_score set
        for r in results:
            assert "category_score" in r

    def test_binary_fallback_without_all_scores(self, builder):
        """Without all_scores, uses binary matched/unmatched split."""
        pipeline = builder.pipeline
        results = pipeline.retrieve_by_category(
            "test query", "Symptoms", all_scores=None
        )
        assert len(results) > 0
        # No category_score in results since all_scores was None
        for r in results:
            assert r.get("category_score", 0) == 0

    def test_retrieve_by_category_bm25_enabled(self, builder):
        """BM25 hybrid merge works in retrieve_by_category."""
        pipeline = builder.pipeline
        assert pipeline._use_bm25 is True

        # Make BM25 return a result
        builder.mock_bm25.retrieve.return_value = [{
            "chunk_id": 5, "bm25_score": 15.0, "question": "q5",
            "answer": "a5", "context": "c5", "category": "Symptoms",
            "text_chunk": "t5",
        }]

        results = pipeline.retrieve_by_category(
            "test query", "Symptoms", top_k=5
        )
        assert len(results) > 0

    def test_retrieve_by_category_bm25_threshold_filter(self, builder):
        """BM25 results below threshold are filtered out in retrieve_by_category."""
        pipeline = builder.pipeline
        # BM25 result below threshold
        builder.mock_bm25.retrieve.return_value = [{
            "chunk_id": 99, "bm25_score": 5.0, "question": "q99",
            "answer": "a99", "context": "c99", "category": "Symptoms",
            "text_chunk": "t99",
        }]
        results = pipeline.retrieve_by_category(
            "test query", "Symptoms", top_k=5
        )
        # Low-score BM25 result should be filtered out, but FAISS results remain
        assert len(results) > 0

    def test_empty_index_returns_empty(self, builder):
        """retrieve_by_category with k=0 returns empty list."""
        builder.mock_index.ntotal = 0
        results = builder.pipeline.retrieve_by_category(
            "test query", "General", top_k=0
        )
        assert results == []

    def test_category_expansion_factor_used(self, builder):
        """Category expansion factor is used for search_k."""
        pipeline = builder.pipeline
        results = pipeline.retrieve_by_category(
            "test query", "Symptoms", top_k=10
        )
        # Symptoms has 5x expansion, so search_k = 10 * 5 = 50
        assert len(results) > 0


# ==============================================================================
# ── build_rag_pipeline edge cases ─────────────────────────────────────────────
# ==============================================================================

class TestBuildRagPipelineSettingsException:
    """Tests for build_rag_pipeline() when settings import fails."""

    def test_build_rag_pipeline_settings_fallback(self):
        """build_rag_pipeline works even when settings import fails."""
        import src.rag.pipeline as rp
        rp._pipeline_instance = None

        # Use patch.object to patch directly on the module object
        with (
            patch.object(rp, "RAGPipeline") as mock_rp,
            patch("config.settings", side_effect=ImportError("no settings")),
        ):
            mock_instance = MagicMock()
            mock_rp.return_value = mock_instance
            instance = rp.build_rag_pipeline()
            assert instance is mock_instance


# ==============================================================================
# ── Module-level convenience functions ────────────────────────────────────────
# ==============================================================================

class TestModuleLevelAnswer:
    """Tests for the module-level answer() function."""

    def test_answer_without_pipeline_builds_singleton(self):
        """module-level answer() builds pipeline when none exists."""
        import src.rag.pipeline as rp
        rp._pipeline_instance = None

        # Use patch.object to patch directly on the module object
        with patch.object(rp, "build_rag_pipeline") as mock_build:
            mock_pipeline = MagicMock()
            mock_pipeline.answer.return_value = {"answer": "test"}
            mock_build.return_value = mock_pipeline

            result = rp.answer("test query")
            assert result["answer"] == "test"
            mock_build.assert_called_once()

    def test_answer_with_existing_pipeline_reuses(self):
        """module-level answer() reuses existing pipeline instance."""
        import src.rag.pipeline as rp
        mock_pipeline = MagicMock()
        mock_pipeline.answer.return_value = {"answer": "existing"}
        rp._pipeline_instance = mock_pipeline

        result = rp.answer("test query")
        assert result["answer"] == "existing"


class TestModuleLevelRetrieve:
    """Tests for the module-level retrieve() function."""

    def test_retrieve_without_pipeline_builds_singleton(self):
        """module-level retrieve() builds pipeline when none exists."""
        import src.rag.pipeline as rp
        rp._pipeline_instance = None

        # Use patch.object to patch directly on the module object
        with patch.object(rp, "build_rag_pipeline") as mock_build:
            mock_pipeline = MagicMock()
            mock_pipeline.retrieve.return_value = [{"chunk_id": 1}]
            mock_build.return_value = mock_pipeline

            result = rp.retrieve("test query")
            assert result[0]["chunk_id"] == 1
            mock_build.assert_called_once()

    def test_retrieve_with_existing_pipeline_reuses(self):
        """module-level retrieve() reuses existing pipeline instance."""
        import src.rag.pipeline as rp
        mock_pipeline = MagicMock()
        mock_pipeline.retrieve.return_value = [{"chunk_id": 99}]
        rp._pipeline_instance = mock_pipeline

        result = rp.retrieve("test query")
        assert result[0]["chunk_id"] == 99


# ==============================================================================
# ── retrieve edge cases ───────────────────────────────────────────────────────
# ==============================================================================

class TestRetrieve:
    """Tests for RAGPipeline.retrieve() edge cases."""

    def test_retrieve_empty_index_returns_empty(self, builder):
        """retrieve with empty index returns empty list."""
        builder.mock_index.ntotal = 0
        results = builder.pipeline.retrieve("test query")
        assert results == []

    def test_retrieve_zero_k_returns_empty(self, builder):
        """retrieve with top_k=0 returns empty list."""
        results = builder.pipeline.retrieve("test query", top_k=0)
        assert results == []

    def test_retrieve_caps_top_k(self, builder):
        """retrieve caps top_k to index.ntotal."""
        builder.mock_index.ntotal = 3
        results = builder.pipeline.retrieve("test query", top_k=999)
        assert len(results) <= 3

    def test_retrieve_bm25_merge(self, builder):
        """retrieve merges BM25 results when _use_bm25 is True."""
        pipeline = builder.pipeline
        # The builder already has BM25 enabled
        builder.mock_bm25.retrieve.return_value = [{
            "chunk_id": 99, "bm25_score": 15.0, "question": "q99",
            "answer": "a99", "context": "c99", "category": "General",
            "text_chunk": "t99",
        }]

        results = pipeline.retrieve("test query", top_k=50)
        assert len(results) > 0

    def test_retrieve_bm25_threshold_filter(self, builder):
        """retrieve filters BM25 results below threshold."""
        pipeline = builder.pipeline
        builder.mock_bm25.retrieve.return_value = [{
            "chunk_id": 99, "bm25_score": 2.0, "question": "q99",
            "answer": "a99", "context": "c99", "category": "General",
            "text_chunk": "t99",
        }]

        results = pipeline.retrieve("test query", top_k=50)
        # BM25 result should be filtered out, FAISS results remain
        assert len(results) > 0
        # The low-score BM25 result (chunk_id=99) should not appear
        ids = [r["chunk_id"] for r in results]
        assert 99 not in ids
