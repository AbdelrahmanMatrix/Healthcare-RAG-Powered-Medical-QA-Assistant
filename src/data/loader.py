"""
Data loading utilities for the Healthcare RAG pipeline.

Handles loading raw, cleaned, and labelled datasets from local disk,
and downloading from HuggingFace datasets.

Public API:
    load_raw_data(path)         → pd.DataFrame
    load_cleaned_data(path)     → pd.DataFrame
    load_labelled_data(path)    → pd.DataFrame
    load_dataset_from_hub()     → pd.DataFrame (raw PubMedQA)
"""

import os
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ── Default paths ─────────────────────────────────────────────────────────────

DEFAULT_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "pubmedqa_raw.csv"
DEFAULT_CLEANED_PATH = PROJECT_ROOT / "data" / "processed" / "pubmedqa_cleaned.csv"
DEFAULT_LABELLED_PATH = PROJECT_ROOT / "data" / "processed" / "pubmedqa_labelled.csv"


def load_raw_data(path: str = None) -> pd.DataFrame:
    """
    Load raw PubMedQA data from CSV.

    Args:
        path: Path to raw CSV. Defaults to data/raw/pubmedqa_raw.csv.

    Returns:
        DataFrame with columns: pubid, question, context, long_answer, final_decision.
    """
    p = path or str(DEFAULT_RAW_PATH)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"Raw data not found at {p}. Run `python download.py` or "
            f"notebooks/01_data_loading.ipynb first."
        )
    return pd.read_csv(p)


def load_cleaned_data(path: str = None) -> pd.DataFrame:
    """
    Load cleaned PubMedQA data from CSV.

    Args:
        path: Path to cleaned CSV. Defaults to data/processed/pubmedqa_cleaned.csv.

    Returns:
        DataFrame with cleaned question, context, answer columns.
    """
    p = path or str(DEFAULT_CLEANED_PATH)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"Cleaned data not found at {p}. Run notebooks/02_preprocessing.ipynb first."
        )
    return pd.read_csv(p)


def load_labelled_data(path: str = None) -> pd.DataFrame:
    """
    Load labelled PubMedQA data from CSV.

    Args:
        path: Path to labelled CSV. Defaults to data/processed/pubmedqa_labelled.csv.

    Returns:
        DataFrame with question, context, answer, category columns.
    """
    p = path or str(DEFAULT_LABELLED_PATH)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"Labelled data not found at {p}. Run notebooks/03_category_labelling.ipynb first."
        )
    return pd.read_csv(p)


def load_dataset_from_hub(split: str = "train", subset: str = "pqa_artificial") -> pd.DataFrame:
    """
    Load raw PubMedQA dataset directly from HuggingFace datasets hub.

    This is the canonical starting point for the pipeline.
    Equivalent to notebook 01's first cell.

    Args:
        split: Dataset split ("train", "validation", "test").
        subset: PubMedQA subset ("pqa_artificial", "pqa_labeled", "pqa_unlabeled").

    Returns:
        DataFrame with raw PubMedQA records.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets package required. Run: pip install datasets")

    ds = load_dataset("qiaojin/PubMedQA", subset, split=split, trust_remote_code=True)
    df = ds.to_pandas()
    return df
