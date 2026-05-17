from pydantic import BaseModel, Field
from typing import List, Optional


class QueryRequest(BaseModel):
    question: str = Field(
        ..., min_length=5, max_length=1000,
        example="What are the symptoms of type 2 diabetes?"
    )
    top_k: Optional[int] = Field(
        default=None, ge=1, le=20,
        description="Override the default number of retrieved chunks (1–20). "
                    "Omit to use the pipeline default (5)."
    )
    category: Optional[str] = Field(
        default=None,
        description=(
            "Force retrieval to prioritise this medical category. "
            "Valid values: Symptoms, Diagnosis, Treatment, Medication, Prevention, General. "
            "Omit to let the classifier infer it automatically."
        ),
        example="Treatment"
    )


class SourceCitation(BaseModel):
    chunk_id:  str
    question:  str    # the PubMed question associated with this chunk
    category:  str
    distance:  float  # L2 distance — lower means more similar


class QueryResponse(BaseModel):
    answer:             str
    category:           str
    retrieved_sources:  List[str]           # chunk IDs as strings (backward compat)
    source_citations:   List[SourceCitation] = []  # richer citations (new)
    disclaimer:         str                 # always injected from config.settings


class HealthResponse(BaseModel):
    status:       str  = "ok"
    model_loaded: bool = False
