# config/settings.py
# Centralised configuration for all pipeline components.
# Values can be overridden by environment variables via .env

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    # Paths (relative to repo root)
    FAISS_INDEX_PATH: str  = "data/embeddings/faiss_index/pubmedqa_index_flatl2.faiss"
    CHUNKS_PKL_PATH: str   = "data/embeddings/faiss_index/chunk_mapping.pkl"
    EVAL_HOLDOUT_PATH: str = "data/processed/eval_holdout.csv"

    # Classifier — upgraded from DistilBERT → BioBERT
    CLASSIFIER_PATH: str    = "models/classifier/biobert_classifier"
    HF_CLASSIFIER_REPO: str = "AbdoMatrix/biobert-medical-classifier"

    # ── Embedding model ────────────────────────────────────────────────────
    # Upgraded from all-MiniLM-L6-v2 (general-purpose) to PubMed-domain model.
    EMBEDDING_MODEL: str = "pritamdeka/S-PubMedBert-MS-MARCO"

    # ── LLM ───────────────────────────────────────────────────────────────
    LLM_MODEL: str    = "llama-3.1-8b-instant"
    LLM_BASE_URL: str = "https://api.groq.com/openai/v1"

    # ── Reranker ──────────────────────────────────────────────────────────
    USE_RERANKER: bool  = True
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── Pipeline behaviour ─────────────────────────────────────────────────
    TOP_K: int    = 10
    INJECT_K: int = 3
    MAX_TOKENS: int = 512
    BM25_THRESHOLD: float = 5.0

    # ── API ───────────────────────────────────────────────────────────────
    CORS_ORIGINS: list = ["*"]
    API_KEY: str       = ""
    GROQ_API_KEY: str  = ""

    # ── Deployment ────────────────────────────────────────────────────────
    DEPLOY_ENV: str  = "local"
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
        extra="ignore",
    )


settings = Settings()
