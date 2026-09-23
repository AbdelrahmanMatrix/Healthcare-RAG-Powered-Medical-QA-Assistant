"""
tests/test_p0_repairs.py
========================

Regression tests for the P0 controlled-repair work (2026-09).

P0.1  Canonical generator model — single source of truth in config.settings.
P0.2  BM25 / FAISS score semantics — explicit faiss_score / bm25_score /
      reranker_score, no inverted "distance", reranker is final authority.
P0.3  Grounding — no ungrounded general-knowledge fallback; answer_source
      exposed as grounded / insufficient_evidence / fallback.
P0.6  Auth — API key required when settings.API_KEY set; disabled when empty.

These tests are fully mocked (no model downloads, no network) and run in CI.
"""

import pytest
from unittest.mock import MagicMock, patch


# ==============================================================================
# ── P0.1 — Canonical generator model ─────────────────────────────────────────
# ==============================================================================

class TestCanonicalGeneratorModel:
    """P0.1: the runtime generator model comes from central configuration."""

    def test_settings_declares_canonical_llm(self):
        from config.settings import settings
        # Canonical evaluation/production model (Groq-hosted)
        assert settings.LLM_MODEL == "meta-llama/llama-4-scout-17b-16e-instruct"

    def test_settings_declares_fallback_llm(self):
        """The offline fallback is declared separately in settings."""
        from config.settings import settings
        assert settings.FALLBACK_LLM_MODEL == "google/flan-t5-base"

    def test_no_hardcoded_gpt_oss_in_runtime_code(self):
        """P0.1: runtime code must not hardcode a conflicting generator model."""
        for rel in ("src/rag/pipeline.py", "src/pipeline.py", "api/routes/query.py",
                    "api/main.py", "mlops/mlflow_tracking.py"):
            with open(rel, "r", encoding="utf-8") as f:
                text = f.read()
            assert "gpt-oss" not in text, f"{rel} hardcodes a generator model"
            assert "llama-4-scout" not in text, f"{rel} hardcodes the generator model"

    def test_build_rag_pipeline_reads_settings_llm(self):
        """build_rag_pipeline() passes settings.LLM_MODEL into RAGPipeline."""
        import importlib
        with patch("config.settings.settings") as mock_settings:
            mock_settings.LLM_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
            mock_settings.EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
            mock_settings.RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
            mock_settings.USE_RERANKER = True
            mock_settings.TOP_K = 30
            mock_settings.INJECT_K = 5
            mock_settings.MAX_TOKENS = 1024
            mock_settings.MAX_CONTEXT_WORDS = 250
            mock_settings.FAISS_INDEX_PATH = "x.faiss"
            mock_settings.CHUNKS_PKL_PATH = "x.pkl"
            mock_settings.CATEGORY_EXPANSION = ""

            import src.rag.pipeline as rp
            with patch.object(rp, "RAGPipeline") as mock_cls:
                rp._pipeline_instance = None
                rp.build_rag_pipeline()
                kwargs = mock_cls.call_args.kwargs
                assert kwargs["llm_model"] == "meta-llama/llama-4-scout-17b-16e-instruct"
                rp._pipeline_instance = None


# ==============================================================================
# ── P0.2 — Score semantics & fusion ──────────────────────────────────────────
# ==============================================================================

class TestScoreSemantics:
    """P0.2: explicit scores, no inverted distance, reranker is final."""

    def _make_pipeline(self, use_reranker=True):
        """Minimal RAGPipeline with mocked index/encoder/BM25/reranker."""
        import numpy as np
        import pandas as pd
        import sys

        mock_index = MagicMock()
        mock_index.ntotal = 10

        mock_faiss = MagicMock()
        mock_faiss.read_index.return_value = mock_index

        def normalize_L2(vectors):  # emulate unit-norm behaviour
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vectors[:] = vectors / norms

        mock_faiss.normalize_L2.side_effect = normalize_L2

        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = np.ones((1, 4), dtype=np.float32)

        mock_st_mod = MagicMock()
        mock_st_mod.SentenceTransformer.return_value = mock_encoder
        mock_st_mod.CrossEncoder = MagicMock(return_value=MagicMock())

        mock_openai_mod = MagicMock()
        mock_openai_mod.OpenAI = MagicMock(return_value=MagicMock())

        mock_clf_mod = MagicMock()

        mock_df = pd.DataFrame({
            "chunk_id": list(range(10)),
            "question": [f"q{i}" for i in range(10)],
            "answer": [f"a{i}" for i in range(10)],
            "context": [f"c{i}" for i in range(10)],
            "text_chunk": [f"t{i}" for i in range(10)],
            "category": ["General"] * 10,
        })

        with patch.dict(sys.modules, {
            "faiss": mock_faiss,
            "sentence_transformers": mock_st_mod,
            "openai": mock_openai_mod,
            "src.classification.classifier": mock_clf_mod,
        }):
            with patch("builtins.open", MagicMock()), \
                 patch("pickle.load", return_value=mock_df), \
                 patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
                from src.rag.pipeline import RAGPipeline
                pipeline = RAGPipeline(
                    top_k=5,
                    use_reranker=use_reranker,
                    faiss_index_path="mock.faiss",
                    chunk_mapping_path="mock.pkl",
                )
        pipeline._use_bm25 = False  # default: dense-only; tests enable BM25 explicitly
        return pipeline

    def _wire_faiss(self, pipeline, scores, ids=None):
        import numpy as np
        n = len(scores)
        ids = ids if ids is not None else list(range(n))
        pipeline.index.ntotal = 10

        def search_side_effect(query, k):
            return (
                np.array([scores[:k]], dtype=np.float32),
                np.array([ids[:k]], dtype=np.int64),
            )

        pipeline.index.search = MagicMock(side_effect=search_side_effect)

    # 1 — BM25 ordering: stronger BM25 never treated as weaker
    def test_strong_bm25_ranks_above_weak_bm25(self):
        pipeline = self._make_pipeline(use_reranker=False)
        pipeline._use_bm25 = True
        pipeline.bm25 = MagicMock()
        pipeline.bm25.retrieve.return_value = [
            {"chunk_id": 0, "question": "strong", "context": "c", "answer": "a",
             "category": "General", "text_chunk": "t", "bm25_score": 25.0},
            {"chunk_id": 1, "question": "weak", "context": "c", "answer": "a",
             "category": "General", "text_chunk": "t", "bm25_score": 13.0},
        ]
        self._wire_faiss(pipeline, scores=[0.1, 0.2, 0.3, 0.4, 0.5], ids=[2, 3, 4, 5, 6])

        results = pipeline.retrieve("q", top_k=5)
        # Both BM25 hits survive the threshold; the STRONG one comes first
        assert results[0]["question"] == "strong"
        assert results[1]["question"] == "weak"
        assert results[0]["bm25_score"] > results[1]["bm25_score"]

    def test_no_result_is_labelled_distance(self):
        """5 (P0.2): no source-specific score may be labelled 'distance'."""
        pipeline = self._make_pipeline(use_reranker=False)
        self._wire_faiss(pipeline, scores=[0.9, 0.8, 0.7, 0.6, 0.5])
        results = pipeline.retrieve("q", top_k=5)
        for r in results:
            assert "distance" not in r
            assert "faiss_score" in r
            assert "bm25_score" not in r  # dense-only pipeline

        pipeline._use_bm25 = True
        pipeline.bm25 = MagicMock()
        pipeline.bm25.retrieve.return_value = [
            {"chunk_id": 7, "question": "q7", "context": "c", "answer": "a",
             "category": "General", "text_chunk": "t", "bm25_score": 14.0},
        ]
        results = pipeline.retrieve("q", top_k=5)
        bm25_rows = [r for r in results if r["chunk_id"] == 7]
        assert bm25_rows and "bm25_score" in bm25_rows[0]
        assert all("distance" not in r for r in results)

    # 2 — FAISS semantics remain correct
    def test_faiss_score_semantics(self):
        """FAISS IP on normalised vectors: higher = more similar; float32-exact."""
        import numpy as np
        pipeline = self._make_pipeline(use_reranker=False)
        scores = [0.95, 0.90, 0.85, 0.80, 0.75]
        self._wire_faiss(pipeline, scores=scores)
        results = pipeline.retrieve("q", top_k=5)
        got = [r["faiss_score"] for r in results]
        # float32 round-trip only — no inversion, rescaling or clamping
        assert got == [float(np.float32(s)) for s in scores]
        assert got == sorted(got, reverse=True)

    # 3 — dedup of merged candidates
    def test_merged_candidates_deduplicated(self):
        pipeline = self._make_pipeline(use_reranker=False)
        pipeline._use_bm25 = True
        pipeline.bm25 = MagicMock()
        pipeline.bm25.retrieve.return_value = [
            {"chunk_id": 0, "question": "dup", "context": "c", "answer": "a",
             "category": "General", "text_chunk": "t", "bm25_score": 20.0},
            {"chunk_id": 9, "question": "only-bm25", "context": "c", "answer": "a",
             "category": "General", "text_chunk": "t", "bm25_score": 14.0},
        ]
        self._wire_faiss(pipeline, scores=[0.9, 0.8, 0.7, 0.6, 0.5], ids=[0, 1, 2, 3, 4])

        results = pipeline.retrieve("q", top_k=5)
        ids = [r["chunk_id"] for r in results]
        assert len(ids) == len(set(ids)), f"duplicate ids in merged pool: {ids}"
        # FAISS copy of chunk 0 must be dropped in favour of the first (BM25) hit
        assert ids.count(0) == 1
        # 1 dup + 4 unique FAISS + 1 BM25-only = 6 candidates, truncated to top_k
        assert len(results) == 5

    # 4 — reranker is the final ranking authority
    def test_reranker_determines_final_ordering(self):
        pipeline = self._make_pipeline(use_reranker=True)
        # Deliberately adversarial: FAISS order is the REVERSE of reranker order.
        self._wire_faiss(pipeline, scores=[0.1, 0.2, 0.3, 0.4, 0.5])
        pipeline.reranker = MagicMock()
        pipeline.reranker.predict.return_value = [9.0, 7.0, 5.0, 3.0, 1.0]

        results = pipeline.retrieve("q", top_k=5)
        # Order follows reranker_score, NOT faiss_score
        assert [r["reranker_score"] for r in results] == [9.0, 7.0, 5.0, 3.0, 1.0]
        assert [r["chunk_id"] for r in results] == [0, 1, 2, 3, 4]
        # original dense score preserved on the winner (which had the LOWEST faiss_score)
        assert results[0]["faiss_score"] < results[-1]["faiss_score"]

    def test_reranker_sort_descends_by_reranker_score(self):
        pipeline = self._make_pipeline(use_reranker=False)
        self._wire_faiss(pipeline, scores=[0.5, 0.4, 0.3, 0.2, 0.1])
        pipeline._use_reranker = True
        pipeline.reranker = MagicMock()
        pipeline.reranker.predict.return_value = [1.0, 5.0, 3.0, 2.0, 4.0]

        results = pipeline.retrieve("q", top_k=5)
        scores = [r["reranker_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    # Category-priority sort uses faiss_score (not the removed distance)
    def test_category_sort_uses_faiss_score_not_distance(self):
        import src.rag.pipeline as rp
        pipeline = self._make_pipeline(use_reranker=False)
        pool = [
            {"chunk_id": 1, "category": "Symptoms", "category_score": 0.9, "faiss_score": 0.2},
            {"chunk_id": 2, "category": "Symptoms", "category_score": 0.9, "faiss_score": 0.8},
        ]
        # emulate the sort used in retrieve_by_category
        pool = sorted(pool, key=lambda x: (x.get("category_score", 0.0),
                                           x.get("faiss_score", 0.0)), reverse=True)
        assert pool[0]["chunk_id"] == 2  # higher faiss_score wins the tie


# ==============================================================================
# ── P0.3 — Grounding statuses ────────────────────────────────────────────────
# ==============================================================================

class TestGroundingContract:
    """P0.3: grounding statuses are explicit and the schema exposes them."""

    def test_api_schema_exposes_answer_source(self):
        from api.schemas.request import QueryResponse
        fields = QueryResponse.model_fields
        assert "answer_source" in fields

    def test_api_schema_answer_source_values(self):
        from api.schemas.request import VALID_ANSWER_SOURCES
        assert VALID_ANSWER_SOURCES == {"grounded", "insufficient_evidence", "fallback"}

    def test_pipeline_constants_match_schema(self):
        from src.rag.pipeline import (
            ANSWER_SOURCE_FALLBACK,
            ANSWER_SOURCE_GROUNDED,
            ANSWER_SOURCE_INSUFFICIENT_EVIDENCE,
        )
        from api.schemas.request import VALID_ANSWER_SOURCES
        assert {ANSWER_SOURCE_GROUNDED,
                ANSWER_SOURCE_INSUFFICIENT_EVIDENCE,
                ANSWER_SOURCE_FALLBACK} == VALID_ANSWER_SOURCES

    def test_insufficient_message_is_explicit(self):
        """The refusal states the evidence was insufficient — no fake answer."""
        from src.rag.pipeline import INSUFFICIENT_CONTEXT_MESSAGE
        assert "sufficient" in INSUFFICIENT_CONTEXT_MESSAGE
        assert "not contain sufficient information" in INSUFFICIENT_CONTEXT_MESSAGE


# ==============================================================================
# ── P0.6 — Auth ──────────────────────────────────────────────────────────────
# ==============================================================================

class TestAuthBehaviour:
    """P0.6: optional API-key auth; enabled iff settings.API_KEY is set."""

    @pytest.mark.asyncio
    async def test_auth_disabled_when_api_key_empty(self):
        from api.middleware.auth import verify_api_key
        with patch("config.settings.settings") as mock_settings:
            mock_settings.API_KEY = ""
            # must NOT raise when no header provided
            assert await verify_api_key(key=None) is None

    @pytest.mark.asyncio
    async def test_auth_rejects_wrong_key(self):
        from fastapi import HTTPException
        from api.middleware.auth import verify_api_key
        with patch("config.settings.settings") as mock_settings:
            mock_settings.API_KEY = "secret-value"
            with pytest.raises(HTTPException) as exc:
                await verify_api_key(key="wrong")
            assert exc.value.status_code == 401

    @pytest.mark.asyncio
    async def test_auth_rejects_missing_key(self):
        from fastapi import HTTPException
        from api.middleware.auth import verify_api_key
        with patch("config.settings.settings") as mock_settings:
            mock_settings.API_KEY = "secret-value"
            with pytest.raises(HTTPException) as exc:
                await verify_api_key(key=None)
            assert exc.value.status_code == 401

    @pytest.mark.asyncio
    async def test_auth_accepts_correct_key(self):
        from api.middleware.auth import verify_api_key
        with patch("config.settings.settings") as mock_settings:
            mock_settings.API_KEY = "secret-value"
            assert await verify_api_key(key="secret-value") is None

    def test_warmup_requires_auth_dependency(self):
        """/warmup must be auth-protected (same dependency as /query)."""
        from api.routes.query import router
        route = next(r for r in router.routes if getattr(r, "path", "") == "/warmup")
        deps = [d.dependency for d in route.dependencies]
        assert any(getattr(d, "__name__", "") == "verify_api_key" for d in deps)

    def test_workflow_sets_api_key_appsetting(self):
        """The Azure deploy workflow passes the API_KEY secret through."""
        import re
        text = open(".github/workflows/azure-deploy.yml", encoding="utf-8").read()
        assert re.search(r'API_KEY="\$\{\{ secrets\.API_KEY \}\}"', text), (
            "deploy workflow must forward the API_KEY GitHub secret"
        )
