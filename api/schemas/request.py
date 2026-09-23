from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import List, Optional

VALID_CATEGORIES = {
    "Symptoms",
    "Diagnosis",
    "Treatment",
    "Medication",
    "Prevention",
    "General",
}

# Grounding statuses exposed to API consumers (mirrors src/rag/pipeline.py).
VALID_ANSWER_SOURCES = {"grounded", "insufficient_evidence", "fallback"}


class QueryRequest(BaseModel):
    question: str = Field(
        ..., min_length=5, max_length=1000,
        json_schema_extra={"example": "What are the symptoms of type 2 diabetes?"}
    )
    top_k: Optional[int] = Field(
        default=None, ge=1, le=30,
        description="Override the default number of retrieved chunks (1–30).",
    )
    category: Optional[str] = Field(
        default=None,
        description=(
            "Force retrieval to prioritise this medical category. "
            "Valid values: Symptoms, Diagnosis, Treatment, Medication, Prevention, General."
        ),
        json_schema_extra={"example": "Treatment"},
    )

    @field_validator("category")
    @classmethod
    def validate_category(cls, value: Optional[str]) -> Optional[str]:
        if value is None:  # pragma: no cover — no test passes category=None explicitly
            return None  # pragma: no cover

        value = value.strip()
        if not value:
            return None

        normalised = value.title()
        if normalised not in VALID_CATEGORIES:
            valid = ", ".join(sorted(VALID_CATEGORIES))
            raise ValueError(f"category must be one of: {valid}")
        return normalised


class SourceCitation(BaseModel):  # pragma: no cover — class def; coverage.py doesn't count module-level class lines
    chunk_id: str
    question: str
    category: str
    # Explicit, per-source ranking scores (P0.2 repair). `distance` /
    # `relevance_score` were removed: BM25 and FAISS scores live on
    # incomparable scales and were previously conflated into one mislabelled
    # "distance" field. The list order IS the final relevance ranking.
    faiss_score: Optional[float] = None   # inner product on L2-normalised vectors (cosine sim; higher = better)
    bm25_score: Optional[float] = None    # BM25 relevance (unbounded; higher = better)
    reranker_score: Optional[float] = None  # CrossEncoder logit (final ranking authority)
    excerpt: str = ""              # first 150 chars of retrieved context


class QueryResponse(BaseModel):
    answer: str
    category: str
    retrieved_sources: List[str]
    source_citations: List[SourceCitation] = Field(default_factory=list)
    # Grounding status (P0.3 repair): "grounded" (answer from retrieved
    # evidence), "insufficient_evidence" (explicit refusal — the retrieved
    # evidence did not support an answer), or "fallback" (best-chunk /
    # error fallback was used).
    answer_source: str = "grounded"
    disclaimer: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str = "ok"
    model_loaded: bool = False
    classifier_ready: bool = False
    groq_configured: bool = False
    index_vectors: int = 0
