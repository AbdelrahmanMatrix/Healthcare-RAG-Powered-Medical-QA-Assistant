from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional


class QueryRequest(BaseModel):
    question: str = Field(
        ..., min_length=5, max_length=1000,
        json_schema_extra={"example": "What are the symptoms of type 2 diabetes?"}
    )
    top_k: Optional[int] = Field(
        default=None, ge=1, le=20,
        description="Override the default number of retrieved chunks (1–20).",
    )
    category: Optional[str] = Field(
        default=None,
        description=(
            "Force retrieval to prioritise this medical category. "
            "Valid values: Symptoms, Diagnosis, Treatment, Medication, Prevention, General."
        ),
        json_schema_extra={"example": "Treatment"},
    )


class SourceCitation(BaseModel):
    chunk_id: str
    question: str
    category: str
    distance: float
    relevance_score: float = 0.0   # 0-1, higher = more relevant (normalised)
    excerpt: str = ""              # first 150 chars of retrieved context


class QueryResponse(BaseModel):
    answer: str
    category: str
    retrieved_sources: List[str]
    source_citations: List[SourceCitation] = []
    disclaimer: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str = "ok"
    model_loaded: bool = False
    classifier_ready: bool = False
    groq_configured: bool = False
    index_vectors: int = 0
