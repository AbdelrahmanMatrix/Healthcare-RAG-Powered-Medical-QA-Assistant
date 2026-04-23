# tests/test_rag_pipeline.py
# Basic smoke tests for the integrated RAG + classifier pipeline

import pytest
from src.rag.pipeline import retrieve, answer
from src.classification.classifier import predict

# --- Classifier tests ---

def test_classifier_returns_valid_category():
    """Classifier should return one of the 6 valid categories."""
    valid = {"Symptoms", "Diagnosis", "Treatment", "Medication", "Prevention", "General"}
    result = predict("What are the symptoms of diabetes?")
    assert result in valid

def test_classifier_returns_string():
    result = predict("How is pneumonia treated?")
    assert isinstance(result, str)
    assert len(result) > 0

# --- RAG pipeline tests ---

def test_retrieve_returns_list():
    """FAISS retrieval should return a non-empty list."""
    results = retrieve("What causes high blood pressure?", k=5)
    assert isinstance(results, list)
    assert len(results) == 5

def test_answer_contains_disclaimer():
    """Every answer must contain the medical disclaimer."""
    result = answer("What is the treatment for type 2 diabetes?")
    assert result["disclaimer_present"] is True
    assert "DISCLAIMER" in result["answer"] or "educational" in result["answer"].lower()

def test_answer_returns_expected_keys():
    """Answer dict should have required keys for the API."""
    result = answer("What are symptoms of hypertension?")
    assert "answer" in result
    assert "retrieved_sources" in result
    assert "disclaimer_present" in result