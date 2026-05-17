# tests/test_rag_pipeline.py
# Integration tests for the RAG + classifier pipeline.
# Requires real FAISS index + model files.
#
# Run in CI:        skipped automatically (no models available)
# Run locally:      pytest tests/test_rag_pipeline.py -v -m integration
# Run all locally:  pytest tests/ -v

import pytest

pytestmark = pytest.mark.integration  # skip in CI unless explicitly requested


@pytest.fixture(scope="module")
def rag_pipeline():
    """Load the real RAG pipeline once for the whole module."""
    from src.rag.pipeline import build_rag_pipeline
    return build_rag_pipeline()


@pytest.fixture(scope="module")
def classifier():
    from src.classification.classifier import load_classifier
    return load_classifier()


def test_classifier_returns_valid_category(classifier):
    """Classifier should return one of the 6 valid categories."""
    valid = {"Symptoms", "Diagnosis", "Treatment", "Medication", "Prevention", "General"}
    result = classifier.predict("What are the symptoms of diabetes?")
    assert result in valid


def test_classifier_returns_string(classifier):
    result = classifier.predict("How is pneumonia treated?")
    assert isinstance(result, str) and len(result) > 0


def test_retrieve_returns_list(rag_pipeline):
    """FAISS retrieval should return a non-empty list of the right length."""
    results = rag_pipeline.retrieve("What causes high blood pressure?", top_k=5)
    assert isinstance(results, list)
    assert len(results) == 5


def test_answer_contains_disclaimer(rag_pipeline):
    """Every answer must have disclaimer_present=True."""
    result = rag_pipeline.answer("What is the treatment for type 2 diabetes?")
    assert result["disclaimer_present"] is True
    assert "DISCLAIMER" in result["answer"] or "educational" in result["answer"].lower()


def test_answer_returns_expected_keys(rag_pipeline):
    """Answer dict must have all keys required by the API layer."""
    result = rag_pipeline.answer("What are symptoms of hypertension?")
    for key in ("answer", "answer_raw", "retrieved_sources", "disclaimer_present", "top_k"):
        assert key in result, f"Missing key: {key}"
