from transformers import pipeline
from typing import Dict

class MedicalQueryClassifier:
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        # In a real scenario, we'd use a medical-specific classifier or fine-tuned DistilBERT
        # For this demo, we'll use a zero-shot classifier to categorize queries
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        self.categories = ["Symptoms", "Diagnosis", "Treatment", "Medication", "General Health"]

    def classify(self, query: str) -> Dict[str, any]:
        result = self.classifier(query, candidate_labels=self.categories)
        return {
            "category": result["labels"][0],
            "confidence": result["scores"][0],
            "all_scores": dict(zip(result["labels"], result["scores"]))
        }

if __name__ == "__main__":
    classifier = MedicalQueryClassifier()
    test_query = "What are the side effects of ibuprofen?"
    prediction = classifier.classify(test_query)
    print(f"Query: {test_query}")
    print(f"Predicted Category: {prediction['category']} (Confidence: {prediction['confidence']:.2f})")
