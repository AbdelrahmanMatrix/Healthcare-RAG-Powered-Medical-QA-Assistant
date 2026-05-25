"""
Medical category labeller for the Healthcare RAG pipeline.

Assigns one of 6 medical categories to a question:
    Symptoms, Diagnosis, Treatment, Medication, Prevention, General

Uses keyword-based pattern matching — lightweight, deterministic,
and fast enough to label 200k+ rows in minutes.

Public API:
    assign_category(question_text) → str  (one of the 6 categories)
"""

import re

# ── Category keyword lists (ordered by specificity — first match wins) ────────

_SYMPTOMS = {
    "symptom", "symptoms", "pain", "fever", "fatigue", "cough", "headache",
    "nausea", "dizziness", "rash", "swelling", "bleeding", "inflammation",
    "itching", "numbness", "shortness of breath", "chest pain", "back pain",
    "weight loss", "weight gain", "loss of appetite", "night sweats",
    "chills", "vomiting", "diarrhoea", "diarrhea", "constipation",
    "sore throat", "runny nose", "congestion", "blurred vision",
    "hearing loss", "tinnitus", "insomnia", "anxiety", "depression",
    "palpitations", "sweating", "tremor", "seizure", "convulsion",
    "jaundice", "pallor", "flushing", "weakness", "cramps",
    "symptomatic", "clinical feature", "manifestation", "presentation",
    "signs of", "indicate", "warning sign",
}

_DIAGNOSIS = {
    "diagnos", "diagnosis", "diagnostic", "diagnose", "test", "testing",
    "screen", "screening", "biopsy", "scan", "x-ray", "mri", "ct scan",
    "ultrasound", "blood test", "lab test", "laboratory", "pathology",
    "detect", "detection", "identify", "identification", "examination",
    "assessment", "evaluation", "staging", "differentiate",
    "differential", "rule out", "predictor", "biomarker",
    "how is ... diagnosed", "diagnostic criteria", "diagnostic test",
    "accuracy of", "sensitivity", "specificity",
}

_TREATMENT = {
    "treatment", "treat", "therapy", "therapeutic", "surgery", "surgical",
    "operation", "transplant", "implant", "procedure", "intervention",
    "management of", "care", "cure", "healing", "recovery",
    "rehabilitation", "physiotherapy", "radiation", "radiotherapy",
    "chemotherapy", "dialysis", "oxygen", "ventilation",
    "resection", "bypass", "angioplasty", "endoscopy",
    "how is ... treated", "treatment option", "treatment for",
    "available treatment", "effectiveness of", "efficacy of",
    "outcome of", "prognosis", "survival", "remission",
}

_MEDICATION = {
    "medication", "medicine", "drug", "pharmaceutical", "prescription",
    "dose", "dosage", "tablet", "capsule", "injection", "infusion",
    "antibiotic", "antiviral", "antifungal", "vaccine", "vaccination",
    "side effect", "adverse effect", "interaction", "contraindication",
    "overdose", "withdrawal", "addiction", "tolerance",
    "pharmacology", "pharmacokinetic", "pharmacodynamic",
    "administration", "oral", "intravenous", "topical", "inhaled",
    "common side effects of", "uses of", "is it safe to take",
    "ibuprofen", "paracetamol", "acetaminophen", "metformin",
    "aspirin", "statin", "opioid", "corticosteroid", "steroid",
    "antidepressant", "antipsychotic", "antihypertensive",
    "chemotherapy drug", "immunosuppressant",
}

_PREVENTION = {
    "prevent", "prevention", "preventive", "prophylaxis", "prophylactic",
    "risk reduction", "reduce risk", "lifestyle change", "diet",
    "exercise", "nutrition", "vaccination", "vaccine", "immunisation",
    "immunization", "healthy", "wellness", "avoid", "avoidance",
    "protective", "protection against", "how to prevent",
    "how can i reduce", "ways to prevent", "stop smoking",
    "quit smoking", "alcohol moderation", "weight management",
    "sunscreen", "hand hygiene", "safe sex", "screening",
    "early detection", "primary prevention", "secondary prevention",
    "how effective are", "how effective is",
}

_GENERAL = {
    "what is", "define", "definition", "explain", "overview",
    "introduction", "basics of", "difference between",
    "how does", "how do", "what does", "meaning of",
    "description of", "understanding", "example of",
    "mechanism", "pathophysiology", "etiology",
}


# ── Combined patterns (compiled once at module load) ──────────────────────────

def _build_pattern(keywords: set) -> re.Pattern:
    """Build a case-insensitive regex pattern from a set of keywords."""
    # Sort by length (longest first) to match multi-word phrases before single words
    sorted_kw = sorted(keywords, key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(re.escape(kw) for kw in sorted_kw) + r")\b", re.IGNORECASE)


_PATTERNS = {
    "Symptoms":   _build_pattern(_SYMPTOMS),
    "Diagnosis":  _build_pattern(_DIAGNOSIS),
    "Treatment":  _build_pattern(_TREATMENT),
    "Medication": _build_pattern(_MEDICATION),
    "Prevention": _build_pattern(_PREVENTION),
    "General":    _build_pattern(_GENERAL),
}

CATEGORY_ORDER = [
    "Symptoms",
    "Diagnosis",
    "Treatment",
    "Medication",
    "Prevention",
    "General",
]


def assign_category(question: str) -> str:
    """
    Assign a medical category to a question using keyword matching.

    Rules:
      - First category with a keyword match wins (order: Symptoms →
        Diagnosis → Treatment → Medication → Prevention).
      - "General" is the fallback (matched last).
      - Matching is case-insensitive and word-boundary-aware.

    Args:
        question: A medical question string.

    Returns:
        One of: Symptoms, Diagnosis, Treatment, Medication, Prevention, General.
    """
    if not question or not isinstance(question, str):
        return "General"

    question = question.strip()

    for cat in CATEGORY_ORDER:
        pattern = _PATTERNS[cat]
        if pattern.search(question):
            return cat

    return "General"


def assign_category_batch(questions: list) -> list:
    """
    Assign categories to a list of questions.

    Args:
        questions: List of question strings.

    Returns:
        List of category strings (same order as input).
    """
    return [assign_category(q) for q in questions]
