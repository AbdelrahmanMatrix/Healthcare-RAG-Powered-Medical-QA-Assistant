"""
Text preprocessing utilities for the Healthcare RAG pipeline.

Transforms raw PubMedQA data into cleaned, normalised text ready for
embedding generation and category labelling.

Public API:
    clean_text(text)        → str  (lowercase, remove HTML, collapse spaces)
    extract_question(text)  → str  (extract question from PubMedQA format)
    extract_context(text)   → str  (extract context from PubMedQA format)
"""

import re
import ast

# ── Patterns ──────────────────────────────────────────────────────────────────
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTISPACE_RE = re.compile(r"\s+")
_NON_ASCII_RE = re.compile(r"[^\x00-\x7F]+")

# PubMedQA context comes as a dict-string with a "contexts" key (list of sentences)
_CONTEXTS_PATTERN = re.compile(r"['\"]contexts['\"]\s*:\s*\[([^\]]+)\]", re.DOTALL)


def clean_text(text: str, remove_non_ascii: bool = False) -> str:
    """
    Normalise text for embedding generation.

    Steps:
      1. Lowercase
      2. Strip HTML tags
      3. Collapse multiple spaces / newlines into single space
      4. Strip leading/trailing whitespace
      5. Optionally remove non-ASCII characters

    Args:
        text: Raw input string.
        remove_non_ascii: If True, strips characters outside ASCII range
                          (useful for keyword-based methods like BM25).

    Returns:
        Cleaned, normalised string.
    """
    if not text or not isinstance(text, str):
        return ""

    text = text.lower()
    text = _HTML_TAG_RE.sub(" ", text)
    text = _MULTISPACE_RE.sub(" ", text).strip()

    if remove_non_ascii:
        text = _NON_ASCII_RE.sub("", text).strip()

    return text


def extract_question(text: str) -> str:
    """
    Extract a question from a raw PubMedQA record.

    Handles:
      - Plain question strings
      - "Question: ..." prefixed formats (from CSV re-exports)
      - Dict-string formats that embed question fields

    Args:
        text: Raw text that may contain a question.

    Returns:
        Cleaned question string, or empty string if no question found.
    """
    if not text or not isinstance(text, str):
        return ""

    text = text.strip()

    # Direct question — already clean
    if text.startswith("Question:"):
        return text[len("Question:"):].strip()

    # Try to parse as a dict-string (some PubMedQA exports)
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, dict):
            for key in ("question", "Question"):
                if key in parsed and parsed[key]:
                    return str(parsed[key]).strip()
    except (ValueError, SyntaxError, MemoryError):
        pass

    # Fallback: return as-is
    return text


def extract_context(text: str) -> str:
    """
    Extract context text from a raw PubMedQA record.

    PubMedQA stores contexts as a dictionary string:
        {'contexts': ['sentence 1.', 'sentence 2.'], 'labels': [...], 'meshes': [...]}

    This function:
      1. Parses the dict-string
      2. Extracts and joins the 'contexts' list
      3. Strips label/mesh metadata (does not leak into retrieval)

    Args:
        text: Raw context field from PubMedQA (usually a dict-string).

    Returns:
        Cleaned, joined context string, or the original text if parsing fails.
    """
    if not text or not isinstance(text, str):
        return ""

    text = text.strip()

    # Try to parse as dict-string
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, dict):
            contexts = parsed.get("contexts", [])
            if contexts and isinstance(contexts, list):
                return " ".join(str(c).strip() for c in contexts if c)
    except (ValueError, SyntaxError, MemoryError):
        pass

    # Regex fallback: extract from 'contexts': [...]
    m = _CONTEXTS_PATTERN.search(text)
    if m:
        raw_list = m.group(1)
        # Split on commas at the top level (crude but effective for PubMedQA)
        parts = re.findall(r"['\"]([^'\"]+)['\"]", raw_list)
        if parts:
            return " ".join(p.strip() for p in parts)

    # Plain text fallback
    if text.startswith("context:"):
        return text[len("context:"):].strip()

    return text
