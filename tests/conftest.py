# tests/conftest.py
# Shared pytest fixtures for API + pipeline testing

import pytest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from api.main import app


# =========================================================
# 🔹 Mock Data
# =========================================================
MOCK_RESULT = {
    "answer": "Mock answer.",
    "category": "Symptoms",
    "sources": ["42"]
}


# =========================================================
# 🔹 API Fixtures
# =========================================================
@pytest.fixture
def api_client():
    """FastAPI test client for endpoint testing."""
    return TestClient(app)


@pytest.fixture
def mock_pipeline():
    """
    Mock successful pipeline execution.
    Replaces the real RAG pipeline with a fixed response.
    """
    with patch("api.routes.query.run_pipeline") as mock:
        mock.return_value = MOCK_RESULT
        yield mock


@pytest.fixture
def mock_pipeline_error():
    """
    Mock pipeline failure.
    Simulates runtime errors inside the pipeline.
    """
    with patch("api.routes.query.run_pipeline") as mock:
        mock.side_effect = RuntimeError("Simulated failure")
        yield mock


@pytest.fixture
def sample_question():
    """Sample medical question used in API tests."""
    return "What are the main symptoms of type 2 diabetes?"


# =========================================================
# 🔹 Project-Level Fixtures (KEEP EXISTING)
# =========================================================
@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Return the absolute path to the repository root."""
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def sample_queries() -> list[str]:
    """A small set of diverse medical queries for pipeline smoke tests."""
    return [
        "What are the symptoms of diabetes?",
        "How is pneumonia diagnosed?",
        "What is the treatment for hypertension?",
        "What are the side effects of metformin?",
        "How can cardiovascular disease be prevented?",
    ]