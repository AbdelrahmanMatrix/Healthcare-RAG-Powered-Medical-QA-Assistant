from pydantic import BaseModel, Field
from typing import List, Optional


class QueryRequest(BaseModel):
    question: str = Field(
        ..., min_length=5, max_length=1000,
        example="What are the symptoms of type 2 diabetes?"
    )


class QueryResponse(BaseModel):
    answer:            str
    category:          str
    retrieved_sources: List[str]
    disclaimer:        str   # always present — injected from config.settings


class HealthResponse(BaseModel):
    status:       str            = "ok"
    model_loaded: Optional[bool] = None