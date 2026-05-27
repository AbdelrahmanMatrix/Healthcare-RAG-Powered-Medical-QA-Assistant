"""
tests/test_integration_full_pipeline.py
=========================================

End-to-end integration tests that exercise the **real** RAG pipeline with
live models (FAISS index, PubMedBERT embeddings, BioBERT classifier,
CrossEncoder reranker, Groq LLM or flan-t5-base fallback).

These tests are **not mocked** — they verify that all components load,
interoperate, and produce correct outputs together.

Marked ``integration`` → skipped automatically in CI (no models available).
Run locally with:

    pytest tests/test_integration_full_pipeline.py -v -m integration

To run **all** integration tests (including the existing test_rag_pipeline.py):

    pytest tests/ -v -m integration
"""

import pytest

pytestmark = pytest.mark.integration


# ==============================================================================
# ── Session-scoped fixtures (loaded once per test session) ───────────────────
# ==============================================================================

@pytest.fixture(scope="session")
def rag_pipeline():
    """Build the real RAG pipeline once per session (thread-safe singleton).

    Loads the FAISS index, embedding model, BM25, reranker, and LLM.
    Uses the cached singleton from build_rag_pipeline() so the pipeline
    is only initialised once across all integration tests.
    """
    from src.rag.pipeline import build_rag_pipeline
    return build_rag_pipeline()


@pytest.fixture(scope="session")
def classifier():
    """Load the real BioBERT classifier once per session.

    Downloads from HuggingFace if not cached locally.
    """
    from src.classification.classifier import load_classifier
    return load_classifier()


@pytest.fixture(scope="session")
def sample_medical_queries() -> list[dict]:
    """Diverse medical queries spanning all 6 categories for routing tests."""
    return [
        {"question": "What are the symptoms of type 2 diabetes?",        "expected_category": "Symptoms"},
        {"question": "How is pneumonia diagnosed?",                      "expected_category": "Diagnosis"},
        {"question": "What is the treatment for hypertension?",          "expected_category": "Treatment"},
        {"question": "What are the side effects of metformin?",          "expected_category": "Medication"},
        {"question": "How can cardiovascular disease be prevented?",      "expected_category": "Prevention"},
        {"question": "What is HbA1c and why is it important?",           "expected_category": "General"},
    ]


# ==============================================================================
# ── Classifier Integration Tests ─────────────────────────────────────────────
# ==============================================================================

class TestClassifierIntegration:
    """Verify the real BioBERT classifier works end-to-end."""

    def test_classifier_returns_valid_category(self, classifier, sample_medical_queries):
        """Every medical query maps to one of the 6 valid categories."""
        valid_categories = {"Symptoms", "Diagnosis", "Treatment", "Medication", "Prevention", "General"}
        for item in sample_medical_queries:
            result = classifier.predict(item["question"])
            assert result in valid_categories, (
                f"'{item['question']}' returned '{result}', "
                f"expected one of {valid_categories}"
            )

    def test_classifier_returns_string(self, classifier):
        """Prediction output is a non-empty string."""
        result = classifier.predict("What causes chronic kidney disease?")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_classifier_confidence_high_for_medical(self, classifier):
        """Medical questions should have high confidence scores (≥ 0.70)."""
        result = classifier.predict_with_confidence("What are the side effects of ibuprofen?")
        assert result["confidence"] >= 0.70, (
            f"Expected confidence ≥ 0.70, got {result['confidence']:.4f}"
        )
        assert result["category"] in classifier.id2label.values()
        assert isinstance(result["all_scores"], dict)
        assert len(result["all_scores"]) == 6  # one probability per category

    def test_classifier_confidence_for_greeting(self, classifier):
        """Greeting/meta queries should have low or high confidence depending on training.

        This test is informational — the classifier is trained on medical QA,
        so it will force-fit greetings into one of the 6 medical categories.
        """
        result = classifier.predict_with_confidence("Hello, how are you?")
        # The result should always produce a valid category + confidence
        assert result["category"] in classifier.id2label.values()
        assert 0.0 <= result["confidence"] <= 1.0

    def test_classifier_batch(self, classifier, sample_medical_queries):
        """Batch prediction works correctly."""
        questions = [item["question"] for item in sample_medical_queries]
        results = classifier.predict_batch(questions)
        assert len(results) == len(questions)
        assert all(isinstance(r, str) for r in results)


# ==============================================================================
# ── Retrieval Integration Tests ───────────────────────────────────────────────
# ==============================================================================

class TestRetrievalIntegration:
    """Verify FAISS + BM25 hybrid retrieval works end-to-end."""

    def test_retrieve_returns_list(self, rag_pipeline):
        """Basic retrieval returns a non-empty list of the right length."""
        results = rag_pipeline.retrieve("What causes hypertension?", top_k=5)
        assert isinstance(results, list)
        assert len(results) == 5

    def test_retrieve_results_have_expected_keys(self, rag_pipeline):
        """Each retrieved chunk has all keys required by the API layer."""
        results = rag_pipeline.retrieve("What are the symptoms of diabetes?", top_k=3)
        assert len(results) > 0
        for r in results:
            assert "chunk_id" in r
            assert "question" in r
            assert "context" in r
            assert "answer" in r
            assert "category" in r
            assert "distance" in r
            assert "text_chunk" in r

    def test_retrieve_sorted_by_distance(self, rag_pipeline):
        """Results are sorted by distance descending (cosine similarity)."""
        results = rag_pipeline.retrieve("What causes hypertension?", top_k=10)
        distances = [r["distance"] for r in results]
        assert distances == sorted(distances, reverse=True), (
            "FAISS IndexFlatIP results should be sorted by distance descending"
        )

    def test_retrieve_with_reranker(self, rag_pipeline):
        """When reranker is enabled, results have reranker_score key."""
        # Ensure reranker is available
        if not getattr(rag_pipeline, "_use_reranker", False):
            pytest.skip("Reranker not available in this pipeline instance")
        results = rag_pipeline.retrieve("What causes hypertension?", top_k=5)
        for r in results:
            assert "reranker_score" in r, (
                f"Chunk {r['chunk_id']} missing reranker_score"
            )

    def test_retrieve_with_bm25_hybrid(self, rag_pipeline):
        """BM25 hybrid results are prepended to FAISS results."""
        if not getattr(rag_pipeline, "_use_bm25", False):
            pytest.skip("BM25 not available in this pipeline instance")
        results = rag_pipeline.retrieve("What causes hypertension?", top_k=10)
        # BM25 results should have bm25_score key
        bm25_results = [r for r in results if "bm25_score" in r]
        # At least some results should have a bm25_score (hybrid path adds it)
        if bm25_results:
            for r in bm25_results:
                assert r["bm25_score"] > rag_pipeline._bm25_threshold, (
                    f"BM25 result {r['chunk_id']} has score {r['bm25_score']:.1f} "
                    f"below threshold {rag_pipeline._bm25_threshold}"
                )

    def test_retrieve_by_category_routing(self, rag_pipeline):
        """Category-routed retrieval returns more results of the target category.

        For the 'Symptoms' category, at least 1 result should be Symptoms.
        """
        results = rag_pipeline.retrieve_by_category(
            "What are the symptoms of diabetes?", "Symptoms", top_k=15
        )
        assert len(results) > 0
        categories = [r["category"] for r in results]
        assert "Symptoms" in categories, (
            "Expected at least one Symptoms result from category-routed retrieval"
        )

    def test_retrieve_by_category_continuous_scoring(self, rag_pipeline, classifier):
        """Category-routed retrieval with continuous scoring includes category_score."""
        result = classifier.predict_with_confidence(
            "What are the symptoms of diabetes?"
        )
        results = rag_pipeline.retrieve_by_category(
            "What are the symptoms of diabetes?",
            result["category"],
            top_k=10,
            all_scores=result["all_scores"],
        )
        assert len(results) > 0
        for r in results:
            assert "category_score" in r
            # category_score should be a valid probability [0, 1]
            assert 0.0 <= r["category_score"] <= 1.0

    def test_retrieve_zero_k(self, rag_pipeline):
        """top_k=0 returns empty list."""
        results = rag_pipeline.retrieve("test query", top_k=0)
        assert results == []

    def test_retrieve_large_k_capped(self, rag_pipeline):
        """top_k larger than index is capped to ntotal."""
        results = rag_pipeline.retrieve("test query", top_k=999_999)
        assert len(results) <= rag_pipeline.index.ntotal

    def test_retrieve_preserves_chunk_order_for_same_distance(self, rag_pipeline):
        """Chunks with similar distances are all returned (no dedup issues)."""
        results = rag_pipeline.retrieve("What causes hypertension?", top_k=15)
        chunk_ids = [r["chunk_id"] for r in results]
        assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunk_ids found"


# ==============================================================================
# ── Generation Integration Tests ──────────────────────────────────────────────
# ==============================================================================

class TestGenerationIntegration:
    """Verify LLM generation works end-to-end."""

    def test_generate_returns_non_empty_string(self, rag_pipeline):
        """generate() returns a non-empty string for a valid query."""
        results = rag_pipeline.retrieve("What causes hypertension?", top_k=5)
        answer = rag_pipeline.generate("What causes hypertension?", results)
        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_generate_with_no_results(self, rag_pipeline):
        """generate() with empty results returns insufficient context message."""
        answer = rag_pipeline.generate("test query", [])
        assert isinstance(answer, str)
        assert len(answer) > 0  # should return fallback message

    def test_generate_extractive_mode(self, rag_pipeline):
        """In extractive mode, generate() returns the top chunk's answer directly."""
        if not hasattr(rag_pipeline, "_extractive"):
            pytest.skip("Pipeline does not support extractive mode toggle")
        results = rag_pipeline.retrieve("What causes hypertension?", top_k=3)
        # Save state and temporarily switch to extractive
        original_mode = rag_pipeline._extractive
        rag_pipeline._extractive = True
        answer = rag_pipeline.generate("What causes hypertension?", results)
        rag_pipeline._extractive = original_mode
        assert isinstance(answer, str)
        assert len(answer) > 0


# ==============================================================================
# ── Full Pipeline Answer Tests ───────────────────────────────────────────────
# ==============================================================================

class TestFullPipelineIntegration:
    """End-to-end pipeline tests — real retrieval + real LLM generation."""

    def test_answer_returns_correct_keys(self, rag_pipeline):
        """answer() returns all keys required by the API layer."""
        result = rag_pipeline.answer("What are the symptoms of hypertension?")
        expected_keys = {
            "question", "answer", "answer_raw", "retrieved_sources",
            "disclaimer_present", "top_k", "used_rag",
            "retrieval_quality", "mean_cosine_similarity",
        }
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_answer_contains_disclaimer(self, rag_pipeline):
        """Every answer must have disclaimer_present=True and disclaimer in answer."""
        result = rag_pipeline.answer("What is the treatment for type 2 diabetes?")
        assert result["disclaimer_present"] is True
        assert "DISCLAIMER" in result["answer"] or "disclaimer" in result["answer"].lower()

    def test_answer_answer_raw_is_non_empty(self, rag_pipeline):
        """The raw answer (before disclaimer) is non-empty for medical queries."""
        result = rag_pipeline.answer("How is pneumonia diagnosed?")
        assert isinstance(result["answer_raw"], str)
        assert len(result["answer_raw"]) > 0

    def test_answer_retrieval_quality_is_float(self, rag_pipeline):
        """retrieval_quality (mean reranker logit) and mean_cosine_similarity are floats.

        Note: reranker scores are CrossEncoder raw logits, NOT probabilities,
        so retrieval_quality can be negative for low-relevance results.
        mean_cosine_similarity uses FAISS IP distances which ARE in [0, 1].
        """
        result = rag_pipeline.answer("What is the treatment for hypertension?")
        assert isinstance(result["retrieval_quality"], float)
        # Reranker scores are raw logits — can be negative for non-relevant chunks
        assert isinstance(result["mean_cosine_similarity"], float)
        # Cosine similarity (FAISS IP distance on normalised vectors) is in [0, 1]
        assert 0.0 <= result["mean_cosine_similarity"] <= 1.0

    def test_answer_top_k_matches_retrieved_count(self, rag_pipeline):
        """top_k field matches the number of retrieved sources."""
        result = rag_pipeline.answer("What causes asthma?", top_k=5)
        assert result["top_k"] == len(result["retrieved_sources"])
        assert result["top_k"] <= 5

    def test_answer_without_retrieval_for_greetings(self, rag_pipeline):
        """Greeting queries may skip retrieval (routing guard)."""
        result = rag_pipeline.answer("Hello")
        assert isinstance(result["used_rag"], bool)
        # used_rag may be False for greetings when the routing guard is active

    def test_answer_retrieved_sources_have_expected_structure(self, rag_pipeline):
        """Each source in retrieved_sources has all expected fields."""
        result = rag_pipeline.answer("What are the symptoms of diabetes?", top_k=3)
        assert result["top_k"] > 0, "Expected at least 1 retrieved source"
        for source in result["retrieved_sources"]:
            expected = {
                "chunk_id", "question", "category", "distance",
                "relevance_score", "reranker_score", "context", "answer", "excerpt",
            }
            for key in expected:
                assert key in source, f"Source missing key: {key}"
            assert isinstance(source["chunk_id"], int)
            assert isinstance(source["distance"], float)
            assert isinstance(source["relevance_score"], float)
            assert 0.0 <= source["relevance_score"] <= 1.0


# ==============================================================================
# ── Classifier + Pipeline End-to-End Routing ──────────────────────────────────
# ==============================================================================

class TestRoutingIntegration:
    """Verify classifier + pipeline routing works end-to-end."""

    def test_answer_with_routing_returns_category(self, rag_pipeline, classifier):
        """answer_with_routing returns a valid category for medical queries."""
        result = rag_pipeline.answer_with_routing(
            "What are the symptoms of diabetes?"
        )
        valid_categories = {"Symptoms", "Diagnosis", "Treatment", "Medication", "Prevention", "General", "Unknown"}
        assert result["category"] in valid_categories, (
            f"Unexpected category: {result['category']}"
        )

    def test_answer_with_routing_classifier_confidence(self, rag_pipeline):
        """answer_with_routing returns classifier confidence score."""
        result = rag_pipeline.answer_with_routing(
            "What is the treatment for hypertension?"
        )
        assert "classifier_confidence" in result
        assert 0.0 <= result["classifier_confidence"] <= 1.0

    def test_answer_with_routing_routing_flag(self, rag_pipeline):
        """answer_with_routing indicates whether routing was applied."""
        result = rag_pipeline.answer_with_routing(
            "How is asthma diagnosed?"
        )
        assert "routing_applied" in result
        assert isinstance(result["routing_applied"], bool)

    def test_answer_with_routing_has_category_matched_sources(self, rag_pipeline):
        """answer_with_routing returns count of category-matched sources."""
        result = rag_pipeline.answer_with_routing(
            "What are the side effects of metformin?", category="Medication"
        )
        assert "category_matched_sources" in result
        assert isinstance(result["category_matched_sources"], int)
        assert result["category_matched_sources"] >= 0

    def test_answer_with_routing_all_keys_present(self, rag_pipeline):
        """answer_with_routing returns all expected keys."""
        result = rag_pipeline.answer_with_routing(
            "How can cardiovascular disease be prevented?"
        )
        expected_keys = {
            "question", "category", "classifier_confidence", "routing_applied",
            "answer", "answer_raw", "retrieved_sources", "disclaimer_present",
            "top_k", "category_matched_sources",
        }
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"


# ==============================================================================
# ── Edge Cases ────────────────────────────────────────────────────────────────
# ==============================================================================

class TestEdgeCasesIntegration:
    """Edge cases for the full pipeline."""

    def test_empty_query(self, rag_pipeline):
        """Empty query returns a valid response (fallback message or disclaimer)."""
        result = rag_pipeline.answer("")
        assert "answer" in result
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

    def test_very_long_query(self, rag_pipeline):
        """Very long query is handled without errors."""
        long_query = "What are the symptoms of " + "chronic " * 200 + "disease?"
        result = rag_pipeline.answer(long_query)
        assert "answer" in result
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

    def test_non_medical_query(self, rag_pipeline):
        """Non-medical query returns a valid response."""
        result = rag_pipeline.answer("What is the capital of France?")
        assert "answer" in result
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

    def test_multi_sentence_query(self, rag_pipeline):
        """Multi-sentence medical query returns a valid response."""
        result = rag_pipeline.answer(
            "My father has type 2 diabetes. He takes metformin. "
            "What are the common side effects? Should he be concerned "
            "about hypoglycemia?"
        )
        assert "answer" in result
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

    def test_short_abbreviated_query(self, rag_pipeline):
        """Short/abbreviated medical query returns a valid response."""
        result = rag_pipeline.answer("COPD symptoms?")
        assert "answer" in result
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0


# ==============================================================================
# ── Pipeline Singleton Tests ──────────────────────────────────────────────────
# ==============================================================================

class TestPipelineSingletonIntegration:
    """Verify the pipeline singleton behaviour."""

    def test_build_rag_pipeline_returns_same_instance(self):
        """build_rag_pipeline() returns the same singleton across calls."""
        from src.rag.pipeline import build_rag_pipeline
        p1 = build_rag_pipeline()
        p2 = build_rag_pipeline()
        assert p1 is p2, "build_rag_pipeline() should return the same singleton"

    def test_module_level_answer_uses_singleton(self):
        """The module-level answer() function uses the cached singleton."""
        from src.rag.pipeline import answer, build_rag_pipeline
        pipeline = build_rag_pipeline()
        result = answer("What causes hypertension?", pipeline=pipeline)
        assert "answer" in result
        assert "disclaimer_present" in result


# ==============================================================================
# ── Evaluation Metrics with Real Data ─────────────────────────────────────────
# ==============================================================================

class TestEvaluationWithRealOutputs:
    """Run evaluation metrics on real pipeline outputs.

    These tests generate actual RAG answers and evaluate them — useful for
    regression testing against known quality baselines.
    """

    @pytest.fixture(scope="class")
    def rag_outputs(self, rag_pipeline):
        """Generate real RAG outputs for a small set of queries."""
        import time
        queries = [
            "What are the symptoms of type 2 diabetes?",
            "What is the treatment for hypertension?",
            "How is pneumonia diagnosed?",
        ]
        outputs = []
        contexts = []
        latencies = []
        for q in queries:
            start = time.time()
            result = rag_pipeline.answer(q)
            elapsed = (time.time() - start) * 1000
            outputs.append(result["answer_raw"])
            latencies.append(elapsed)
            ctx = [
                s.get("context", "") + " " + s.get("answer", "")
                for s in result.get("retrieved_sources", [])
            ]
            contexts.append(ctx if ctx else [""])
        return {
            "queries": queries,
            "outputs": outputs,
            "contexts": contexts,
            "latencies": latencies,
        }

    def test_outputs_are_non_empty(self, rag_outputs):
        """All generated answers are non-empty strings."""
        for i, output in enumerate(rag_outputs["outputs"]):
            assert isinstance(output, str), f"Output {i} is not a string"
            assert len(output) > 0, f"Output {i} is empty"

    def test_latency_within_bounds(self, rag_outputs):
        """Real pipeline latency should be reasonable (< 30s per query)."""
        for i, latency in enumerate(rag_outputs["latencies"]):
            assert latency < 30_000, (
                f"Query {i} took {latency:.0f}ms — exceeds 30s threshold"
            )

    def test_bleu_score_computable(self, rag_outputs):
        """BLEU score can be computed on real pipeline outputs."""
        from src.evaluation.metrics import compute_bleu
        # Use outputs as both predictions and references (self-BLEU sanity check)
        bleu = compute_bleu(rag_outputs["outputs"], rag_outputs["outputs"])
        assert isinstance(bleu, float)
        # Identical outputs should have BLEU > 0
        assert bleu > 0.0, "Self-BLEU should be > 0 for identical strings"

    def test_rouge_score_computable(self, rag_outputs):
        """ROUGE-L can be computed on real pipeline outputs."""
        from src.evaluation.metrics import compute_rouge
        rouge = compute_rouge(rag_outputs["outputs"], rag_outputs["outputs"])
        assert isinstance(rouge, float)
        assert rouge > 0.0, "Self-ROUGE should be > 0 for identical strings"

    def test_evaluate_full_computable(self, rag_outputs):
        """evaluate_full() runs without errors on real outputs."""
        from src.evaluation.metrics import evaluate_full
        result = evaluate_full(
            rag_outputs["outputs"],
            rag_outputs["outputs"],  # self-evaluation for sanity check
            contexts=rag_outputs["contexts"],
            label="IntegrationTest",
        )
        expected_keys = {"label", "bleu", "rouge_l", "bertscore_f1", "faithfulness", "n_samples"}
        for key in expected_keys:
            assert key in result, f"Missing key in evaluate_full result: {key}"
        assert result["label"] == "IntegrationTest"
        assert result["n_samples"] == len(rag_outputs["outputs"])


# ==============================================================================
# ── Format Sources Integration ────────────────────────────────────────────────
# ==============================================================================

class TestFormatSourcesIntegration:
    """Verify format_sources() works with real retrieved data."""

    def test_format_sources_structure(self, rag_pipeline):
        """format_sources() returns a list of dicts with expected structure."""
        retrieved = rag_pipeline.retrieve("What causes hypertension?", top_k=3)
        sources = rag_pipeline.format_sources(retrieved)
        assert isinstance(sources, list)
        assert len(sources) == len(retrieved)
        for source in sources:
            assert "chunk_id" in source
            assert "question" in source
            assert "category" in source
            assert "category_score" in source
            assert "distance" in source
            assert "relevance_score" in source
            assert "reranker_score" in source
            assert "context" in source
            assert "answer" in source
            assert "excerpt" in source

    def test_format_sources_relevance_score_bounds(self, rag_pipeline):
        """relevance_score is always in [0, 1]."""
        retrieved = rag_pipeline.retrieve("What causes hypertension?", top_k=5)
        sources = rag_pipeline.format_sources(retrieved)
        for source in sources:
            assert 0.0 <= source["relevance_score"] <= 1.0, (
                f"relevance_score {source['relevance_score']} out of bounds"
            )
            assert 0.0 <= source["distance"] <= 1.0, (
                f"distance {source['distance']} out of bounds"
            )
