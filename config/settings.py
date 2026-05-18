# config/settings.py
# Centralised configuration for all pipeline components.
# Values can be overridden by environment variables via .env

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

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
    BM25_THRESHOLD: float = 5.0   # BM25 scores above this are treated as high-confidence hits

    # API configuration
    CORS_ORIGINS: list = ["*"]    # Tighten to specific origins before public launch

    # API keys (from .env, never hardcoded)
    API_KEY: str = ""             # if empty, auth is disabled (dev mode)
    GROQ_API_KEY: str = ""

    # Deployment info (reflected in Streamlit dashboard)
    DEPLOY_ENV: str = "local"
    DEPLOY_DATE: str = ""

    disclaimer: str = (
        "This is an informational assistant only. "
        "Always consult a qualified healthcare professional for medical decisions. "
        "Do not use this information as a substitute for professional medical advice, "
        "diagnosis, or treatment."
    )

    categories: list = [
        "Symptoms", "Diagnosis", "Treatment",
        "Medication", "Prevention", "General",
    ]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",           # silently drop unknown env vars (e.g. HF_TOKEN, WEBSITE_PORT)
    )


settings = Settings()
