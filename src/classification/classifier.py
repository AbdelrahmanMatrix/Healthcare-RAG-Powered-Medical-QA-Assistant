"""
DistilBERT medical query classifier.

Classifies medical questions into 6 categories:
Symptoms, Diagnosis, Treatment, Medication, Prevention, General

Loading priority:
1. Local model (models/classifier/distilbert_classifier/)
2. HuggingFace Hub (auto-download if local not found)

Usage:
    from src.classification.classifier import load_classifier, predict

    classifier = load_classifier()
    category = predict("What are the symptoms of diabetes?")
"""

import os
from pathlib import Path

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# ── Config ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_LOCAL_PATH = PROJECT_ROOT / "models" / "classifier" / "distilbert_classifier"
HF_REPO_ID = "AbdooMatrix/distilbert-medical-classifier"  # ← change username


class MedicalClassifier:
    """
    Wrapper for the fine-tuned DistilBERT classifier.

    Loads from local path if available, otherwise downloads from HuggingFace.
    """

    def __init__(self, model_path: str = None):
        path = model_path or str(DEFAULT_LOCAL_PATH)

        # Check if local model exists and has actual model files
        local_exists = (
            os.path.exists(path)
            and os.path.isdir(path)
            and any(
                f.endswith(('.bin', '.safetensors'))
                for f in os.listdir(path)
            )
        )

        if local_exists:
            source = path
            print(f"📂 Loading classifier from local: {path}")
        else:
            source = HF_REPO_ID
            print(f"📥 Local model not found. Downloading from HuggingFace: {HF_REPO_ID}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = DistilBertTokenizer.from_pretrained(source)
        self.model = DistilBertForSequenceClassification.from_pretrained(source)
        self.model.to(self.device)
        self.model.eval()

        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id

        print(f"✅ Classifier loaded | Device: {self.device} | Classes: {list(self.id2label.values())}")

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

    def predict_with_confidence(self, text: str) -> dict:
        """Predict category with confidence scores."""
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)[0]
        pred_id = torch.argmax(probs).item()

        return {
            "category": self.id2label[pred_id],
            "confidence": float(probs[pred_id]),
            "all_scores": {
                self.id2label[i]: float(probs[i])
                for i in range(len(probs))
            }
        }

    def predict_batch(self, texts: list[str]) -> list[str]:
        """Predict categories for a list of texts."""
        return [self.predict(t) for t in texts]


# ── Module-level convenience functions ───────────────────────────────────

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