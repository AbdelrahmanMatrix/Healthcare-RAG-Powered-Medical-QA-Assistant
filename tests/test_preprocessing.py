# tests/test_preprocessing.py
# Fixed: function names now match actual preprocessor API
from src.data.preprocessor import clean_text, extract_question, extract_context
from src.data.labeller import assign_category


def test_clean_text_lowercases():
    result = clean_text("Hello WORLD")
    assert result == result.lower()


def test_clean_text_removes_html():
    result = clean_text("<p>Hello</p>")
    assert "<" not in result and ">" not in result


def test_clean_text_collapses_spaces():
    result = clean_text("too   many    spaces")
    assert "  " not in result


def test_extract_question_finds_keyword():
    text = "Question: what causes fever?"
    result = extract_question(text)
    assert "fever" in result


def test_extract_context_finds_text():
    text = "context: the patient has a history of diabetes."
    result = extract_context(text)
    assert "diabetes" in result


def test_extract_context_dict_string():
    """Handle dict string format from CSV re-load of qiaojin/PubMedQA."""
    text = (
        "{'contexts': ['Patient has diabetes.', 'Blood sugar is high.'],"
        " 'labels': ['BACKGROUND'], 'meshes': ['Diabetes']}"
    )
    result = extract_context(text)
    assert "diabetes" in result
    assert "blood sugar" in result.lower()
    assert "BACKGROUND" not in result  # labels should NOT leak
    assert "meshes" not in result  # meshes key should NOT appear


def test_labeller_returns_valid_category():
    valid = {"Symptoms", "Diagnosis", "Treatment", "Medication", "Prevention", "General"}
    assert assign_category("What are the symptoms of flu?") in valid
    assert assign_category("How is cancer diagnosed?") in valid
    assert assign_category("What is 2+2?") == "General"
