"""
Medical category labeller for PubMedQA dataset.

Algorithm (v2):
  - Per-category SCORING (no first-match-wins).
  - QUESTION weighted 3x over context (intent lives in the question).
  - INTENT PHRASES (e.g. "treatment of", "side effects of") add +5 boost.
  - Ties broken by specificity order: Medication > Prevention > Treatment >
    Diagnosis > Symptoms.

Categories (6): Symptoms, Diagnosis, Treatment, Medication, Prevention, General
"""

import re
import pandas as pd
from collections import defaultdict


# ── Keyword dictionary ──────────────────────────────────────────────────────
# Listed specific-to-general so dict iteration order also reflects priority,
# but actual selection uses scoring + tie-break order below.

CATEGORY_KEYWORDS = {
    'Medication': [
        # Drug classes & specific drugs
        'drug', 'medication', 'pharmacol', 'prescription',
        'antibiotic', 'antimicrob', 'antifung', 'antiviral',
        'antidepress', 'antihypertens', 'opioid', 'nsaid',
        'aspirin', 'statin', 'insulin', 'warfarin', 'heparin',
        'corticoster', 'isoniazid', 'metformin', 'ibuprofen',
        'analges', 'sedat', 'anesthes', 'anaesthes',
        # Drug administration
        'dose', 'dosage', 'tablet', 'capsule', 'pill', 'inject',
        'infusion', 'intravenous', 'topical', 'administer',
        # Drug effects
        'side effect', 'adverse effect', 'adverse drug',
        'toxicity', 'overdose', 'contraindic',
        # Pharmacology concepts
        'receptor', 'agonist', 'antagonist', 'inhibitor',
        'serum level', 'plasma level',
        # Vaccines & supplements
        'vaccine', 'vaccinat', 'immuniz', 'supplement', 'steroid',
        'chemotherap', 'chemother',
    ],
    'Prevention': [
        'prevent', 'prophyla', 'avoid',
        'risk reduction', 'risk factor', 'early detection',
        'screening program', 'health promotion', 'public health',
        'epidemiolog',
        'lifestyle', 'diet', 'exercise', 'smoking cessation',
        'obesity', 'weight loss', 'physical activity',
        'breastfeed', 'safe sex', 'hygiene', 'nutriti',
    ],
    'Treatment': [
        'treat', 'therap', 'cure',
        'surgery', 'surgical', 'operation', 'operative',
        'procedure', 'intervention', 'manage',
        'radiation', 'radiotherap',
        'repair', 'reconstruct', 'resect', 'ablation', 'transplant',
        'dialysis', 'transfusion', 'rehabilitat', 'physiotherap',
        'postoperat', 'perioperat',
        # Common surgical suffixes (catches prostatectomy, tonsillectomy, etc.)
        'ectomy', 'otomy', 'plasty', 'ostomy',
    ],
    'Diagnosis': [
        'diagnos', 'screen', 'biopsy',
        'mri', 'ct scan', 'ultrasound', 'x-ray', 'xray',
        'ecg', 'ekg', 'eeg', 'imaging', 'radiograph',
        'marker', 'biomarker',
        'prognos', 'differential',
        'sensitivity', 'specificity', 'accuracy',
    ],
    'Symptoms': [
        'symptom', 'pain', 'ache', 'fever', 'nausea',
        'vomit', 'diarrhea', 'bleed', 'rash', 'swelling',
        'fatigue', 'malaise', 'cough', 'dizz', 'weakness',
        'manifestation', 'discomfort', 'dysfunction',
    ],
}


# ── Intent phrases (high-weight signals from the QUESTION) ──────────────────
# These directly express the user's intent and override ambiguous keywords.

INTENT_PHRASES = {
    'Treatment': [
        r'treatment of', r'treatment for', r'therapy for', r'therapy of',
        r'manag(e|ing|ement) of', r'how to treat', r'how is .{0,40}treated',
        r'best treatment', r'effective treatment', r'surgical option',
        r'surgical management', r'postoperative',
    ],
    'Diagnosis': [
        r'diagnosis of', r'how is .{0,40}diagnosed',
        r'screening for', r'detect(ion|ing) of',
        r'differential diagnos', r'test for',
    ],
    'Medication': [
        r'side effects? of', r'adverse effects? of',
        r'dose of', r'dosage of', r'efficacy of \w+',
        r'use of \w+ in', r'administration of',
        r'pharmacokinetic', r'pharmacodynamic',
    ],
    'Prevention': [
        r'prevention of', r'preventing', r'how to prevent',
        r'reduce(d)? risk', r'risk reduction',
        r'be prevented', r'prevent(ed|ing)? \w+',
    ],
    'Symptoms': [
        r'symptoms? of', r'signs? of', r'manifestations? of',
        r'present(ation|ing) with', r'clinical presentation',
        r'what are the symptoms',
    ],
}


# ── Scoring config ──────────────────────────────────────────────────────────
QUESTION_WEIGHT = 3       # Question keywords count 3x
CONTEXT_WEIGHT = 1        # Context keywords count 1x
INTENT_PHRASE_BOOST = 5   # Each intent phrase matched in question adds +5

# Tie-break order: when scores tie, prefer the more specific category.
TIE_BREAK_ORDER = ['Medication', 'Prevention', 'Treatment', 'Diagnosis', 'Symptoms']


# ── Pre-compile patterns at import time ─────────────────────────────────────
_KEYWORD_PATTERNS = {
    cat: [re.compile(r'\b' + kw, re.IGNORECASE) for kw in kws]
    for cat, kws in CATEGORY_KEYWORDS.items()
}
_INTENT_PATTERNS = {
    cat: [re.compile(p, re.IGNORECASE) for p in patterns]
    for cat, patterns in INTENT_PHRASES.items()
}


def _count_distinct_matches(text: str, patterns: list) -> int:
    """Count how many distinct patterns match the text (each pattern at most once)."""
    return sum(1 for p in patterns if p.search(text))


def assign_category(question: str, context: str = '', verbose: bool = False):
    """
    Assign a medical category using weighted keyword + intent-phrase scoring.

    Parameters
    ----------
    question : str
        The medical question text. Weighted 3x.
    context : str, optional
        The medical context/abstract text. Weighted 1x.
    verbose : bool, optional
        If True, return (category, score_dict) instead of just category.

    Returns
    -------
    str OR (str, dict)
        Category name; if verbose, also the per-category score dict.
    """
    q = (question or '').lower()
    c = (context or '').lower()

    scores = defaultdict(int)

    # 1. Keyword matches (question 3x, context 1x)
    for cat, patterns in _KEYWORD_PATTERNS.items():
        scores[cat] += QUESTION_WEIGHT * _count_distinct_matches(q, patterns)
        scores[cat] += CONTEXT_WEIGHT * _count_distinct_matches(c, patterns)

    # 2. Intent-phrase boost (question only — intent lives in the question)
    for cat, patterns in _INTENT_PATTERNS.items():
        for p in patterns:
            if p.search(q):
                scores[cat] += INTENT_PHRASE_BOOST
                break  # one boost per category, even if multiple phrases match

    # 3. Pick winner
    max_score = max(scores.values()) if scores else 0
    if max_score == 0:
        result = 'General'
    else:
        winners = [cat for cat, s in scores.items() if s == max_score]
        if len(winners) == 1:
            result = winners[0]
        else:
            # Tie-break by specificity preference
            for preferred in TIE_BREAK_ORDER:
                if preferred in winners:
                    result = preferred
                    break
            else:
                result = winners[0]  # fallback (shouldn't happen)

    return (result, dict(scores)) if verbose else result


def label_dataframe(df: pd.DataFrame,
                    question_col: str = 'question',
                    context_col: str = 'context',
                    output_col: str = 'category',
                    verbose: bool = False) -> pd.DataFrame:
    """
    Apply category labelling to an entire DataFrame.

    If verbose=True, also adds a `_scores` column with per-row score dicts
    (useful for debugging / tuning).
    """
    df = df.copy()
    df[question_col] = df[question_col].fillna('').astype(str)
    df[context_col] = df[context_col].fillna('').astype(str)

    if verbose:
        results = df.apply(
            lambda row: assign_category(row[question_col], row[context_col], verbose=True),
            axis=1
        )
        df[output_col] = [r[0] for r in results]
        df['_scores'] = [r[1] for r in results]
    else:
        df[output_col] = df.apply(
            lambda row: assign_category(row[question_col], row[context_col]),
            axis=1
        )

    return df


def print_category_distribution(df: pd.DataFrame,
                                col: str = 'category') -> pd.Series:
    """
    Print and return category distribution with percentages.
    Flags any category below 1% representation.
    """
    counts = df[col].value_counts()
    total = len(df)

    print(f"\n{'Category':<15} {'Count':>7} {'Percentage':>10}")
    print('─' * 35)

    flagged = []
    for cat, count in counts.items():
        pct = count / total * 100
        flag = '  [LOW <1%]' if pct < 1.0 else ''
        print(f"{cat:<15} {count:>7,} {pct:>9.1f}%{flag}")
        if pct < 1.0:
            flagged.append(cat)

    print(f"\nTotal rows: {total:,}")
    print(f"Categories: {len(counts)}")

    if flagged:
        print(f"\nLow-representation categories: {flagged}")
    else:
        print("\nAll categories have >= 1% representation")

    return counts


def explain(question: str, context: str = '') -> str:
    """
    Print a verbose breakdown of the labelling decision for one example.
    Useful for quick keyword tuning during development.
    """
    cat, scores = assign_category(question, context, verbose=True)
    print(f"Q: {question}")
    if context:
        print(f"   (context: {context[:80]}...)")
    print(f"\nPredicted: {cat}")
    print(f"Scores:")
    for c in sorted(scores, key=scores.get, reverse=True):
        if scores[c] > 0:
            print(f"  {c:<12} {scores[c]}")
    return cat


# ── Self-test (run: python -m src.data.labeller) ────────────────────────────
if __name__ == '__main__':
    test_cases = [
        # (question, expected_category, note)
        ("Is naturopathy as effective as conventional therapy for treatment of menopausal symptoms?",
         "Treatment", "was Symptoms before the fix"),
        ("What are the early symptoms of type 2 diabetes?",
         "Symptoms", ""),
        ("How is pneumonia diagnosed in elderly patients?",
         "Diagnosis", ""),
        ("What are the side effects of metformin?",
         "Medication", ""),
        ("How can cardiovascular disease be prevented through lifestyle changes?",
         "Prevention", "was General/wrong before the fix"),
        ("Is laparoscopic radical prostatectomy better than traditional retropubic radical prostatectomy?",
         "Treatment", "was Symptoms before the fix"),
        ("Does bacterial gastroenteritis predispose people to functional gastrointestinal disorders?",
         "General", "no clear keywords - General is correct"),
        ("Efficacy of secondary isoniazid preventive therapy among HIV-infected Southern Africans?",
         "Medication", "isoniazid drug name should win"),
    ]

    print("=" * 80)
    print("LABELLER SELF-TEST")
    print("=" * 80)

    passed = 0
    for q, expected, note in test_cases:
        actual = assign_category(q)
        ok = actual == expected
        passed += int(ok)
        mark = "PASS" if ok else "FAIL"
        print(f"\n[{mark}]  expected={expected:<11}  got={actual}")
        print(f"        {q}")
        if note:
            print(f"        note: {note}")
        if not ok:
            print(f"        scores: {assign_category(q, verbose=True)[1]}")

    print(f"\n{'=' * 80}")
    print(f"Result: {passed}/{len(test_cases)} passed")
    print('=' * 80)