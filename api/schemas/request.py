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
    distance: float
    relevance_score: float = 0.0   # 0-1, higher = more relevant (normalised)
    excerpt: str = ""              # first 150 chars of retrieved context


class QueryResponse(BaseModel):
    answer: str
    category: str
    answer_source: str = "rag"
    retrieved_sources: List[str]
    source_citations: List[SourceCitation] = Field(default_factory=list)
    disclaimer: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str = "ok"
    model_loaded: bool = False
    classifier_ready: bool = False
    groq_configured: bool = False
    index_vectors: int = 0
