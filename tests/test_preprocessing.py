# tests/test_preprocessing.py

from src.data.preprocessor import normalize_text, get_question, get_context
from src.data.labeller import label_question


def test_normalize_text_lowercases():
    result = normalize_text("Hello WORLD")
    assert result == result.lower()


def test_normalize_text_removes_html():
    result = normalize_text("<p>Hello</p>")
    assert "<" not in result
    assert ">" not in result


def test_normalize_text_collapses_spaces():
    result = normalize_text("too   many    spaces")
    assert "  " not in result


def test_get_question_extracts_correctly():
    text = "context: some context. question: what causes fever?"
    result = get_question(text)
    assert "fever" in result


def test_get_context_extracts_correctly():
    text = "context: the patient has a history of diabetes."
    result = get_context(text)
    assert "diabetes" in result



def test_labeller_returns_valid_category():
    valid = {"Symptoms", "Diagnosis", "Treatment", "Medication", "Prevention", "General"}
    assert label_question("What are the symptoms of flu?") in valid
    assert label_question("How is cancer diagnosed?") in valid
    assert label_question("What is 2+2?") == "General"