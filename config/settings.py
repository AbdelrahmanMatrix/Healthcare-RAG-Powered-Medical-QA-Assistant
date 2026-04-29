# config/settings.py
# Centralised configuration for all pipeline components.
# Values can be overridden by environment variables via .env

from pathlib import Path
from pydantic_settings import BaseSettings

# Repo root — one level above this file
REPO_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    # Paths (relative to repo root)
    FAISS_INDEX_PATH: str = "data/embeddings/faiss_index/pubmedqa_index_flatl2.faiss"
    CHUNKS_PKL_PATH: str = "data/embeddings/faiss_index/chunk_mapping.pkl"
    CLASSIFIER_PATH: str = "models/classifier/distilbert_classifier"

    # Model identifiers
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL: str = "llama-3.1-8b-instant"
    LLM_BASE_URL: str = "https://api.groq.com/openai/v1"

    # Pipeline behaviour
    TOP_K: int = 5
    MAX_TOKENS: int = 512

    # API keys (from .env, never hardcoded)
    GROQ_API_KEY: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()