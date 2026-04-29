"""
Evaluation metrics for RAG pipeline.

Computes BLEU, ROUGE-L, and improvement percentages.

Usage:
    from src.evaluation.metrics import compute_bleu, compute_rouge, compute_improvement
"""

import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


def compute_bleu(predictions: list[str], references: list[str]) -> float:
    """
    Compute average BLEU score across all prediction-reference pairs.
    Uses smoothing to handle short sentences.
    """
    smoother = SmoothingFunction().method1
    scores = []

    for pred, ref in zip(predictions, references):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()

        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            scores.append(0.0)
            continue

        score = sentence_bleu(
            [ref_tokens],
            pred_tokens,
            smoothing_function=smoother
        )
        scores.append(score)

    return float(np.mean(scores))


def compute_rouge(predictions: list[str], references: list[str]) -> float:
    """
    Compute average ROUGE-L F1 score across all prediction-reference pairs.
    """
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = []

    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        scores.append(result['rougeL'].fmeasure)

    return float(np.mean(scores))


def compute_improvement(baseline: float, improved: float) -> float:
    """
    Compute percentage improvement from baseline to improved score.
    Returns percentage (e.g. 25.0 means 25% improvement).
    """
    if baseline == 0:
        return float('inf') if improved > 0 else 0.0
    return ((improved - baseline) / baseline) * 100


def evaluate_pair(
    predictions: list[str],
    references: list[str],
    label: str = "Model"
) -> dict:
    """
    Run full evaluation (BLEU + ROUGE-L) and return results dict.
    """
    bleu = compute_bleu(predictions, references)
    rouge = compute_rouge(predictions, references)

    return {
        "label": label,
        "bleu": round(bleu, 4),
        "rouge_l": round(rouge, 4),
        "n_samples": len(predictions),
    }