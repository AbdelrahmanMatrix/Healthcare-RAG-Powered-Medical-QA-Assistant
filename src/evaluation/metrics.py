from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import evaluate

# Load once at module level (important for efficiency)
_rouge = evaluate.load("rouge")


# =========================
# BLEU (Corpus-level)
# =========================
def compute_bleu(predictions: list, references: list) -> float:
    """
    Compute corpus-level BLEU score.

    Args:
        predictions (list[str]): model outputs
        references (list[str]): ground-truth answers

    Returns:
        float: BLEU score
    """
    smoothie = SmoothingFunction().method4

    # BLEU expects tokenized sentences
    tokenized_refs = [[ref.split()] for ref in references]
    tokenized_preds = [pred.split() for pred in predictions]

    return corpus_bleu(tokenized_refs, tokenized_preds, smoothing_function=smoothie)


# =========================
# ROUGE-L (Corpus-level)
# =========================
def compute_rouge(predictions: list, references: list) -> float:
    """
    Compute corpus-level ROUGE-L F1 score.

    Args:
        predictions (list[str])
        references (list[str])

    Returns:
        float: ROUGE-L F1
    """
    result = _rouge.compute(
        predictions=predictions,
        references=references
    )

    return result["rougeL"]


# =========================
# Improvement metric
# =========================
def compute_improvement(baseline: float, improved: float) -> float:
    """
    Compute percentage improvement.

    Args:
        baseline (float)
        improved (float)

    Returns:
        float: percentage improvement
    """
    if baseline == 0:
        return 0.0

    return ((improved - baseline) / baseline) * 100


# =========================
# Optional: per-sample BLEU (for debugging / analysis)
# =========================
def compute_sentence_bleu(reference: str, prediction: str) -> float:
    """
    Sentence-level BLEU (useful for debugging hallucinations).
    """
    smoothie = SmoothingFunction().method4

    ref_tokens = [reference.split()]
    pred_tokens = prediction.split()

    return corpus_bleu([ref_tokens], [pred_tokens], smoothing_function=smoothie)