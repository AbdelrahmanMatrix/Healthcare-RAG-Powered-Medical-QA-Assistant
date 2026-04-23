# src/classification/classifier.py
# Purpose: load the fine-tuned DistilBERT classifier and predict a category
# for any incoming medical query. Called by the integrated pipeline.

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# These must match exactly what Doha used during training
LABEL_MAP = {
    0: "Symptoms",
    1: "Diagnosis",
    2: "Treatment",
    3: "Medication",
    4: "Prevention",
    5: "General"
}

MODEL_PATH = "models/classifier/distilbert_classifier"  # where Doha saved the fine-tuned model

# Load once at module level so the model isn't reloaded on every call
_tokenizer = None
_model = None

def _load():
    """Load model and tokenizer lazily (only once)."""
    global _tokenizer, _model
    if _model is None:
        _tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
        _model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
        _model.eval()  # disable dropout for inference

def predict(query: str) -> str:
    """
    Classify a medical query into one of 6 categories.
    
    Args:
        query: the raw user question
    
    Returns:
        category label string e.g. "Treatment"
    """
    _load()
    
    # Tokenize exactly as Doha did (max_length=256)
    inputs = _tokenizer(
        query,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True
    )
    
    with torch.no_grad():
        outputs = _model(**inputs)
    
    predicted_id = torch.argmax(outputs.logits, dim=1).item()
    return LABEL_MAP[predicted_id]