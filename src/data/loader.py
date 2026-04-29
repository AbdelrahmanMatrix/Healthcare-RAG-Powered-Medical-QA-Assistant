"""
Data loader utility for Healthcare RAG project.

Provides convenience functions to load raw, cleaned, and labelled datasets.
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "pubmedqa_raw.csv"
CLEANED_PATH = PROJECT_ROOT / "data" / "processed" / "pubmedqa_cleaned.csv"
LABELLED_PATH = PROJECT_ROOT / "data" / "processed" / "pubmedqa_labelled.csv"


def load_raw() -> pd.DataFrame:
    """Load raw PubMedQA dataset."""
    return pd.read_csv(RAW_PATH)


def load_cleaned() -> pd.DataFrame:
    """Load cleaned dataset (3 columns: question, context, answer)."""
    return pd.read_csv(CLEANED_PATH)


def load_labelled() -> pd.DataFrame:
    """Load labelled dataset (4 columns: question, context, answer, category)."""
    return pd.read_csv(LABELLED_PATH)