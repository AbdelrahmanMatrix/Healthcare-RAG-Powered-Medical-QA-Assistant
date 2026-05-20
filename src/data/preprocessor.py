"""
Text cleaning & normalisation pipeline for PubMedQA dataset (qiaojin/PubMedQA).

The raw dataset has these columns:
- pubid: int32
- question: string (already clean)
- context: list of strings (PubMed abstract snippets)
- long_answer: string (detailed answer)
- final_decision: string (yes/no/maybe)

Steps applied (in order):
1. Validate schema (expect question, context, long_answer)
2. Extract clean columns (join context list, use question/long_answer directly)
3. Clean text (HTML, special chars, whitespace, lowercase)
4. Remove rows with missing question or answer
5. Remove non-English rows (langdetect)
6. Remove exact duplicate rows

Usage:
    from src.data.preprocessor import run_preprocessing_pipeline
    df_clean, log = run_preprocessing_pipeline(df_raw)
"""

import ast
import re
import pandas as pd
from langdetect import detect, LangDetectException


# ── Individual cleaning functions ────────────────────────────────────────────

def strip_html(text: str) -> str:
    """Remove HTML tags."""
    return re.sub(r'<[^>]+>', '', str(text))


def remove_special_chars(text: str) -> str:
    """Remove characters that aren't alphanumeric, spaces, or basic punctuation."""
    return re.sub(r'[^a-zA-Z0-9\s.,;:?!\-\'\"()/]', '', str(text))


def collapse_whitespace(text: str) -> str:
    """Replace multiple spaces/newlines with a single space."""
    return re.sub(r'\s+', ' ', str(text))


def clean_text(text: str) -> str:
    """
    Full cleaning pipeline for a single text string.
    Order matters: HTML → special chars → whitespace → lowercase → strip.
    """
    text = strip_html(text)
    text = remove_special_chars(text)
    text = collapse_whitespace(text)
    text = text.lower()
    text = text.strip()
    return text


def is_english(text: str) -> bool:
    """
    Detect if text is English using langdetect.
    Returns True if English or if detection fails (keep the row by default).
    """
    try:
        return detect(str(text)) == 'en'
    except LangDetectException:
        return True  # keep row if detection fails (e.g. too short)


# ── Column extraction helpers ────────────────────────────────────────────────

def extract_context(context_data) -> str:
    """
    Extract context text from raw context data.
    
    Handles three formats:
    - List of strings (qiaojin/PubMedQA from datasets lib) — joins with space
    - Dict string ``{"contexts": [...], ...}`` (qiaojin/PubMedQA reloaded from CSV) — parses dict
    - Pre-formatted string with 'context:' prefix (legacy format) — strips prefix
    """
    # Handle list of strings (qiaojin/PubMedQA from datasets library)
    if isinstance(context_data, list):
        return ' '.join(str(c).strip() for c in context_data if c)

    text = str(context_data).strip()

    # Handle dict string (CSV re-load — pandas reads dicts as strings)
    if text.startswith('{'):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, dict) and 'contexts' in parsed:
                contexts = parsed['contexts']
                if isinstance(contexts, list):
                    return ' '.join(str(c).strip() for c in contexts if c)
        except (ValueError, SyntaxError, MemoryError):
            pass

    # Handle pre-formatted string (legacy format)
    match = re.search(r'context:\s*(.*)', text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def extract_question(question_data: str) -> str:
    """Extract question text from raw question data.
    
    Handles both:
    - Already clean question (qiaojin/PubMedQA) — strips only
    - Pre-formatted with 'Question:' prefix (legacy format)
    """
    text = str(question_data).strip()
    match = re.search(r'Question:\s*(.*)', text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


# ── Main pipeline ────────────────────────────────────────────────────────────

def run_preprocessing_pipeline(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Run the full preprocessing pipeline on the raw DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame with columns from qiaojin/PubMedQA:
        - question: str (already clean)
        - context: list of strings (PubMed abstracts)
        - long_answer: str (detailed answer)
        - final_decision: str (yes/no/maybe)

    Returns
    -------
    tuple[pd.DataFrame, dict]
        - Cleaned DataFrame with columns: question, context, answer
        - Log dictionary with before/after counts for each step
    """
    log = {}
    log['raw_rows'] = len(df)

    # ── Step 1: Validate schema ──────────────────────────────────────────
    # Support both old column schema and current qiaojin/PubMedQA schema
    old_required = {'instruction', 'input', 'output'}
    new_required = {'question', 'context', 'long_answer'}
    
    if old_required.issubset(set(df.columns)):
        # Legacy schema (instruction/input/output columns)
        df = df.copy()
        df['question'] = df['input'].apply(extract_question)
        df['context'] = df['instruction'].apply(extract_context)
        df['answer'] = df['output'].astype(str).str.strip()
    elif new_required.issubset(set(df.columns)):
        # New qiaojin/PubMedQA schema
        df = df.copy()
        df['question'] = df['question'].apply(extract_question)
        df['context'] = df['context'].apply(extract_context)
        df['answer'] = df['long_answer'].astype(str).str.strip()
    else:
        missing_old = old_required - set(df.columns)
        missing_new = new_required - set(df.columns)
        raise ValueError(
            f"Unrecognized schema. Missing old columns: {missing_old}, "
            f"Missing new columns: {missing_new}"
        )

    # Keep only the 3 standard columns
    df = df[['question', 'context', 'answer']]
    log['after_extraction'] = len(df)

    # ── Step 3: Clean text (HTML, special chars, whitespace, lowercase) ──
    for col in ['question', 'context', 'answer']:
        df[col] = df[col].apply(clean_text)
    log['after_cleaning'] = len(df)

    # ── Step 4: Remove missing values ────────────────────────────────────
    before = len(df)
    df = df.dropna(subset=['question', 'answer'])
    df = df[(df['question'].str.len() > 0) & (df['answer'].str.len() > 0)]
    log['removed_missing'] = before - len(df)
    log['after_missing'] = len(df)

    # ── Step 5: Remove non-English rows ──────────────────────────────────
    before = len(df)
    english_mask = df.apply(
        lambda row: is_english(row['question'] + ' ' + row['answer']),
        axis=1
    )
    df = df[english_mask]
    log['removed_non_english'] = before - len(df)
    log['after_english_filter'] = len(df)

    # ── Step 6: Remove exact duplicates ──────────────────────────────────
    before = len(df)
    df = df.drop_duplicates()
    log['removed_duplicates'] = before - len(df)
    log['after_duplicates'] = len(df)

    # ── Final ────────────────────────────────────────────────────────────
    df = df.reset_index(drop=True)
    log['final_rows'] = len(df)

    return df, log


def print_pipeline_log(log: dict) -> None:
    """Pretty-print the preprocessing pipeline log."""
    print('\n📊 Preprocessing Pipeline Log')
    print('─' * 50)
    print(f'  Raw rows:              {log["raw_rows"]:,}')
    print(f'  After extraction:      {log["after_extraction"]:,}')
    print(f'  After text cleaning:   {log["after_cleaning"]:,}')
    print(f'  Removed missing:       {log["removed_missing"]:,}')
    print(f'  After missing filter:  {log["after_missing"]:,}')
    print(f'  Removed non-English:   {log["removed_non_english"]:,}')
    print(f'  After English filter:  {log["after_english_filter"]:,}')
    print(f'  Removed duplicates:    {log["removed_duplicates"]:,}')
    print(f'  Final clean rows:      {log["final_rows"]:,}')
    print('─' * 50)
