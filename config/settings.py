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
    # Groq-hosted OpenAI GPT-OSS 120B (production).  Falls back to flan-t5-base locally.
    LLM_MODEL: str = "openai/gpt-oss-120b"
    LLM_BASE_URL: str = "https://api.groq.com/openai/v1"

    # ── Reranker ──────────────────────────────────────────────────────────────
    # 12-layer MiniLM — higher precision than 6-layer variant.
    USE_RERANKER: bool = True
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"

    # ── Pipeline behaviour ─────────────────────────────────────────────────────
    TOP_K: int = 30      # FAISS retrieval candidates
    INJECT_K: int = 5    # chunks fed to LLM after reranking
    MAX_CONTEXT_WORDS: int = 250  # max words per evidence chunk (Finding 7)
    MAX_TOKENS: int = 1024   # max new tokens for LLM generation (was 512; increased for thorough answers)
    # ── BM25 hybrid retrieval ───────────────────────────────────────────
    # BM25_THRESHOLD governs the keyword-match filter: only BM25 results
    # scoring ABOVE this threshold are prepended to the FAISS pool before
    # CrossEncoder reranking.
    #
    # Calibrated by analyzing BM25 score distribution across 15 test queries
    # (top-50 per query = 750 scores):
    #   - Median: 13.21,  P90: 10.76,  Top-5 range: 31–36
    #   - At 5.0: 100% pass (threshold is a no-op — too low)
    #   - At 12.0: ~70% pass (filters weakest 30% of keyword matches)
    #   - At 15.0: ~28% pass (too aggressive — kills short queries like "aspirin")
    #
    # Setting to 12.0 lets through the strongest keyword matches while
    # leaving the reranker to decide on borderline candidates.
    BM25_THRESHOLD: float = 12.0

    # ── Per-category FAISS expansion factors ────────────────────────────
    # JSON dict overrides for CATEGORY_EXPANSION in pipeline.py.
    # Leave empty to use the defaults (calibrated from category distribution).
    # Example: '{"Symptoms": 6, "General": 4}'
    CATEGORY_EXPANSION: str = ""

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
    DASHBOARD_URL: str = "/dashboard"

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
