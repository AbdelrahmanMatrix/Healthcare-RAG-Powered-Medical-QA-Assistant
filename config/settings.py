# config/settings.py
# Centralised configuration for all pipeline components.
# Values can be overridden by environment variables via .env

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    # ── Data Paths (relative to repo root) ────────────────────────────────────
    FAISS_INDEX_PATH: str = "data/embeddings/faiss_index/pubmedqa_index_flatip.faiss"
    CHUNKS_PKL_PATH: str = "data/embeddings/faiss_index/chunk_mapping.pkl"
    EVAL_HOLDOUT_PATH: str = "data/processed/eval_holdout.csv"

    # ── Classifier ────────────────────────────────────────────────────────────
    # BioBERT fine-tuned on 6 medical categories (Symptoms / Diagnosis /
    # Treatment / Medication / Prevention / General).
    CLASSIFIER_PATH: str = "models/classifier/biobert_classifier"
    HF_CLASSIFIER_REPO: str = "AbdoMatrix/biobert-medical-classifier"

    # ── Embedding model ────────────────────────────────────────────────────────
    # PubMedBERT fine-tuned on MS-MARCO — biomedical retrieval specialist.
    EMBEDDING_MODEL: str = "pritamdeka/S-PubMedBert-MS-MARCO"

    # ── LLM ───────────────────────────────────────────────────────────────────
    # Groq-hosted Llama 4 Scout (preview).  Falls back to flan-t5-base locally.
    LLM_MODEL: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    LLM_BASE_URL: str = "https://api.groq.com/openai/v1"

    # ── Reranker ──────────────────────────────────────────────────────────────
    # 12-layer MiniLM — higher precision than 6-layer variant.
    USE_RERANKER: bool = True
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"

    # ── Pipeline behaviour ─────────────────────────────────────────────────────
    TOP_K: int = 20      # FAISS retrieval candidates
    INJECT_K: int = 3    # chunks fed to LLM after reranking
    MAX_CONTEXT_WORDS: int = 200  # max words per evidence chunk (Finding 7)
    MAX_TOKENS: int = 256    # max new tokens for LLM generation
    BM25_THRESHOLD: float = 5.0

    # ── API ───────────────────────────────────────────────────────────────────
    CORS_ORIGINS: list = ["*"]
    API_KEY: str = ""
    GROQ_API_KEY: str = ""

    # ── HuggingFace ───────────────────────────────────────────────────────────
    HF_TOKEN: str = ""

    # ── Deployment ────────────────────────────────────────────────────────────
    DEPLOY_ENV: str = "local"
    DEPLOY_DATE: str = ""
    AZURE_APP_URL: str = ""

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
