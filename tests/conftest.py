# tests/conftest.py
# Shared pytest fixtures available to all test files.

import pytest
from pathlib import Path


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