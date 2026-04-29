"""
Medical category labeller for PubMedQA dataset.
Assigns one of 6 categories based on keyword-regex matching
applied to the question + context columns.

Categories: Symptoms, Diagnosis, Treatment, Medication, Prevention, General
"""

import re
import pandas as pd

# ── Keyword dictionary ──────────────────────────────────────────────────────
# Merged from both previous implementations, expanded for better coverage.
# Each keyword is matched using word-boundary regex (\b) to avoid partial matches
# e.g. "sign" won't match "design"

CATEGORY_KEYWORDS = {
    'Symptoms': [
        'symptom', 'sign', 'pain', 'fever', 'nausea', 'bleed',
        'fatigue', 'vomit', 'diarrhea', 'swelling', 'rash',
        'manifestation', 'impairment', 'discomfort', 'dysfunction'
    ],
    'Diagnosis': [
        'diagnos', 'detect', 'screen', 'test', 'scan', 'biopsy',
        'assess', 'predict', 'marker', 'accuracy', 'imaging',
        'classification', 'identify', 'evaluate', 'prognosis'
    ],
    'Medication': [
        'drug', 'medication', 'dose', 'dosage', 'antibiotic', 'vaccine',
        'pill', 'inject', 'supplement', 'steroid', 'inhibitor',
        'antidepressant', 'chemotherapy', 'isoniazid',
        'pharmacol', 'prescription', 'tablet', 'capsule',
        'infusion', 'analges', 'sedativ', 'anesthes', 'anaesthes',
        'opioid', 'nsaid', 'aspirin', 'statin', 'insulin',
        'warfarin', 'heparin', 'corticoster', 'antihypertens',
        'antimicrob', 'antifung', 'antiviral', 'adverse effect',
        'side effect', 'toxicity', 'overdose', 'contraindic',
        'administer', 'intravenous', 'oral dos', 'topical',
        'receptor', 'agonist', 'antagonist', 'serum level'
    ],
    'Treatment': [
        'treat', 'therapy', 'cure', 'surgery', 'operation',
        'procedure', 'manage', 'intervention', 'radiation',
        'repair', 'reconstruct', 'resect', 'ablation', 'transplant'
    ],
    'Prevention': [
        'prevent', 'protect', 'avoid', 'lifestyle',
        'diet', 'exercise', 'smoking', 'obesity',
        'vaccination', 'screening program', 'risk reduction',
        'prophyla', 'immuniz', 'hygiene', 'safe sex',
        'breastfeed', 'nutriti', 'physical activity',
        'weight loss', 'risk factor', 'early detection',
        'health promotion', 'public health', 'epidemiolog'
    ]
}

# Pre-compile regex patterns for performance (run once at import time)
_COMPILED_PATTERNS = {
    category: [re.compile(r'\b' + kw, re.IGNORECASE) for kw in keywords]
    for category, keywords in CATEGORY_KEYWORDS.items()
}


def assign_category(question: str, context: str = '') -> str:
    """
    Assign a medical category to a single row based on keyword-regex
    matching against the combined question + context text.

    Parameters
    ----------
    question : str
        The medical question text.
    context : str, optional
        The medical context/abstract text.

    Returns
    -------
    str
        One of: Symptoms, Diagnosis, Treatment, Medication, Prevention, General
    """
    combined = f"{question} {context}".lower()

    for category, patterns in _COMPILED_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(combined):
                return category

    return 'General'


def label_dataframe(df: pd.DataFrame,
                    question_col: str = 'question',
                    context_col: str = 'context',
                    output_col: str = 'category') -> pd.DataFrame:
    """
    Apply category labelling to an entire DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain `question_col` and `context_col` columns.
    question_col : str
        Name of the question column.
    context_col : str
        Name of the context column.
    output_col : str
        Name of the new category column to create.

    Returns
    -------
    pd.DataFrame
        Copy of df with the new `output_col` column added.
    """
    df = df.copy()

    df[question_col] = df[question_col].fillna('').astype(str)
    df[context_col] = df[context_col].fillna('').astype(str)

    df[output_col] = df.apply(
        lambda row: assign_category(row[question_col], row[context_col]),
        axis=1
    )

    return df


def print_category_distribution(df: pd.DataFrame,
                                 col: str = 'category') -> pd.Series:
    """
    Print and return category distribution with percentages.
    Also flags any category below 1% representation.
    """
    counts = df[col].value_counts()
    total = len(df)

    print(f"\n{'Category':<15} {'Count':>7} {'Percentage':>10}")
    print('─' * 35)

    flagged = []
    for cat, count in counts.items():
        pct = count / total * 100
        flag = ' ⚠️  < 1%' if pct < 1.0 else ''
        print(f"{cat:<15} {count:>7,} {pct:>9.1f}%{flag}")
        if pct < 1.0:
            flagged.append(cat)

    print(f"\nTotal rows: {total:,}")
    print(f"Categories: {len(counts)}")

    if flagged:
        print(f"\n⚠️  Low-representation categories: {flagged}")
    else:
        print("\n✅ All categories have ≥ 1% representation")

    return counts