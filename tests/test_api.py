"""
M3 API tests — run with: pytest tests/test_api.py -v
All 9 tests must pass before Docker build.
"""
from unittest.mock import patch
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)
MOCK = {"answer": "Mock answer.", "category": "Symptoms", "sources": ["42"]}

VALID_CATEGORIES = {
    "Symptoms", "Diagnosis", "Treatment",
    "Medication", "Prevention", "General",
}


class TestHealth:
    def test_returns_200(self):
        assert client.get("/health").status_code == 200

    def test_status_field_is_ok(self):
        assert client.get("/health").json()["status"] == "ok"

    def test_latency_header_present(self):
        assert "X-Response-Time-Ms" in client.get("/health").headers


class TestQuery:
    @patch("api.routes.query.run_pipeline")
    def test_returns_200(self, m):
        m.return_value = MOCK
        r = client.post("/query", json={"question": "What causes diabetes?"})
        assert r.status_code == 200

    @patch("api.routes.query.run_pipeline")
    def test_all_required_fields_present(self, m):
        m.return_value = MOCK
        data = client.post("/query", json={"question": "How is hypertension treated?"}).json()
        for field in ("answer", "category", "retrieved_sources", "disclaimer"):
            assert field in data, f"Missing field: {field}"

    @patch("api.routes.query.run_pipeline")
    def test_disclaimer_is_non_empty_string(self, m):
        m.return_value = MOCK
        data = client.post("/query", json={"question": "What is metformin used for?"}).json()
        assert isinstance(data["disclaimer"], str) and len(data["disclaimer"]) > 20

    @patch("api.routes.query.run_pipeline")
    def test_category_is_valid(self, m):
        m.return_value = MOCK
        data = client.post("/query", json={"question": "How to prevent stroke?"}).json()
        assert data["category"] in VALID_CATEGORIES

    @patch("api.routes.query.run_pipeline")
    def test_latency_header_present_on_query(self, m):
        m.return_value = MOCK
        r = client.post("/query", json={"question": "What are flu symptoms?"})
        assert "X-Response-Time-Ms" in r.headers

    def test_missing_question_returns_422(self):
        assert client.post("/query", json={}).status_code == 422

    @patch("api.routes.query.run_pipeline")
    def test_pipeline_error_returns_500(self, m):
        m.side_effect = RuntimeError("FAISS index not found")
        r = client.post("/query", json={"question": "What causes anaemia?"})
        assert r.status_code == 500

    @patch("api.routes.query.run_pipeline")
    def test_source_citations_schema(self, m):
        """source_citations must be a list of objects with the SourceCitation fields."""
        m.return_value = MOCK
        data = client.post("/query", json={"question": "What causes anaemia?"}).json()
        assert isinstance(data["source_citations"], list)
        if data["source_citations"]:
            citation = data["source_citations"][0]
            for field in ("chunk_id", "question", "category", "distance"):
                assert field in citation, f"SourceCitation missing field: {field}"

    def test_question_too_short_returns_422(self):
        # min_length=5 on the question field
        assert client.post("/query", json={"question": "hi"}).status_code == 422

    def test_root_returns_200(self):
        assert client.get("/").status_code == 200
