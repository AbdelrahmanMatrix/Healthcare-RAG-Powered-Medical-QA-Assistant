"""
DistilBERT medical query classifier.

Classifies medical questions into 6 categories:
Symptoms, Diagnosis, Treatment, Medication, Prevention, General

Usage:
    from src.classification.classifier import load_classifier, predict

    classifier = load_classifier()
    category = predict("What are the symptoms of diabetes?", classifier=classifier)
"""

from pathlib import Path
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Default model path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "classifier" / "distilbert_classifier"


class MedicalClassifier:
    """Wrapper for the fine-tuned DistilBERT classifier."""

    def __init__(self, model_path: str = None):
        path = model_path or str(DEFAULT_MODEL_PATH)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading classifier from: {path}")
        self.tokenizer = DistilBertTokenizer.from_pretrained(path)
        self.model = DistilBertForSequenceClassification.from_pretrained(path)
        self.model.to(self.device)
        self.model.eval()

        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id
        print(f"✅ Classifier loaded | Classes: {list(self.id2label.values())}")

    def predict(self, text: str) -> str:
        """Predict category for a single text input."""
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        pred_id = torch.argmax(outputs.logits, dim=1).item()
        return self.id2label[pred_id]

    def predict_batch(self, texts: list[str]) -> list[str]:
        """Predict categories for a list of texts."""
        return [self.predict(t) for t in texts]


# ── Module-level convenience functions ───────────────────────────────────────

_classifier_instance = None


def load_classifier(**kwargs) -> MedicalClassifier:
    """Load and cache classifier instance."""
    global _classifier_instance
    _classifier_instance = MedicalClassifier(**kwargs)
    return _classifier_instance


def predict(text: str, classifier: MedicalClassifier = None) -> str:
    """Predict category for a single query."""
    global _classifier_instance
    if classifier is None:
        if _classifier_instance is None:
            _classifier_instance = load_classifier()
        classifier = _classifier_instance
    return classifier.predict(text)