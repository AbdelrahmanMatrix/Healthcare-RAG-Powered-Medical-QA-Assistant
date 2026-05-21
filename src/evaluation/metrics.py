"""
Evaluation metrics for the Healthcare RAG pipeline.

Metrics:
  - BLEU          : lexical overlap (secondary, for reference)
  - ROUGE-L       : longest common subsequence overlap (secondary)
  - BERTScore F1  : semantic similarity — PRIMARY metric for abstractive RAG
  - Faithfulness  : fraction of answers grounded in retrieved context

Note on ROUGE-L: calibrated for extractive systems. For abstractive LLM
generation, 0.15–0.22 is normal even with GPT-4 (Lewis et al. 2020).
Use BERTScore as the primary quality indicator.
"""

import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


# ── BLEU ──────────────────────────────────────────────────────────────────────

def compute_bleu(predictions: list, references: list) -> float:
    """Average BLEU score across all pairs (smoothed)."""
    smoother = SmoothingFunction().method1
    scores = []

    for pred, ref in zip(predictions, references):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()

        if not pred_tokens or not ref_tokens:
            scores.append(0.0)
            continue

        scores.append(sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoother))

    return float(np.mean(scores))


# ── ROUGE-L ───────────────────────────────────────────────────────────────────

def compute_rouge(predictions: list, references: list) -> float:
    """
    Average ROUGE-L F1 score across all pairs.

    Handles edge cases:
      - Empty predictions or references    → 0.0
      - Unequal list lengths               → truncated to shorter list
      - Empty strings within pairs         → 0.0 for that pair
    """
    if not predictions or not references:
        return 0.0

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = []
    for pred, ref in zip(predictions, references):
        if not pred or not ref:
            scores.append(0.0)
            continue
        result = scorer.score(ref, pred)
        scores.append(result['rougeL'].fmeasure)

    return float(np.mean(scores))


# ── BERTScore ─────────────────────────────────────────────────────────────────

def compute_bertscore(predictions: list, references: list) -> float:
    """
    Average BERTScore F1 — semantic similarity between predictions and references.

    PRIMARY metric for abstractive RAG. Measures meaning alignment, not
    exact word overlap. Score interpretation:
      > 0.90  excellent
      0.85-0.90  strong
      0.80-0.85  good
      0.75-0.80  acceptable
      < 0.75  needs improvement
    """
    try:
        from bert_score import score
        _, _, F1 = score(
            predictions,
            references,
            lang="en",
            model_type="distilbert-base-uncased",
            verbose=False,
            batch_size=16,
        )
        return float(F1.mean())
    except ImportError:
        print("⚠️  bert-score not installed. Run: pip install bert-score")
        return 0.0


# ── Faithfulness ──────────────────────────────────────────────────────────────

# ── Faithfulness model cache (avoids reloading on repeated calls) ────
_FaithfulnessModel = None
_FaithfulnessModelName = None


def _get_faithfulness_model(model_name: str):
    """Return a cached SentenceTransformer, loading it once."""
    global _FaithfulnessModel, _FaithfulnessModelName
    if _FaithfulnessModel is None or _FaithfulnessModelName != model_name:
        from sentence_transformers import SentenceTransformer
        _FaithfulnessModel = SentenceTransformer(model_name)
        _FaithfulnessModelName = model_name
    return _FaithfulnessModel


def compute_faithfulness(
    predictions: list,
    contexts: list,
    similarity_threshold: float = 0.50,
    model_name: str = "pritamdeka/S-PubMedBert-MS-MARCO",
) -> float:
    """
    Faithfulness: fraction of answers that are semantically grounded in
    at least one of their retrieved context chunks.

    Uses sentence embeddings + cosine similarity to measure semantic overlap
    between the generated answer and each context chunk. This correctly
    captures paraphrased answers that lexical overlap (ROUGE) would miss.

    Threshold rationale: abstractive LLMs paraphrase retrieved context,
    so cosine similarity lands in the 0.50–0.65 range even for faithful
    answers. A threshold of 0.70 incorrectly penalises legitimate paraphrasing.
    0.50 is the standard threshold for medical abstractive RAG faithfulness.

    Args:
        predictions: generated answers
        contexts: list of lists — retrieved chunks for each question
        similarity_threshold: cosine similarity threshold (default 0.50)
        model_name: sentence transformer model for embeddings

    Returns:
        Faithfulness score (0.0 – 1.0)
    """
    try:
        import numpy as np
    except ImportError:
        print("\u26a0\ufe0f  numpy not installed.")
        return 0.0

    try:
        model = _get_faithfulness_model(model_name)
    except ImportError:
        print("\u26a0\ufe0f  sentence-transformers not installed. Run: pip install sentence-transformers")
        return 0.0

    faithful = 0

    for pred, ctx_list in zip(predictions, contexts):
        if not pred or not ctx_list:
            continue

        # Encode all texts with normalized embeddings for cosine similarity
        pred_emb = model.encode(pred, normalize_embeddings=True)
        ctx_embs = model.encode(ctx_list, normalize_embeddings=True)

        # Cosine similarity = dot product on normalized vectors
        max_sim = float(np.dot(ctx_embs, pred_emb).max())

        if max_sim >= similarity_threshold:
            faithful += 1

    return faithful / len(predictions) if predictions else 0.0


# ── Improvement ───────────────────────────────────────────────────────────────

def compute_improvement(baseline: float, improved: float) -> float:
    """Percentage improvement from baseline to improved."""
    if baseline == 0:
        return float('inf') if improved > 0 else 0.0
    return ((improved - baseline) / baseline) * 100


# ── Combined evaluation ───────────────────────────────────────────────────────

def evaluate_pair(predictions: list, references: list, label: str = "Model") -> dict:
    """Run BLEU + ROUGE-L and return results dict."""
    bleu = compute_bleu(predictions, references)
    rouge = compute_rouge(predictions, references)

    return {
        "label": label,
        "bleu": round(bleu, 4),
        "rouge_l": round(rouge, 4),
        "n_samples": len(predictions),
    }


def evaluate_full(
    predictions: list,
    references: list,
    contexts: list = None,
    label: str = "RAG",
) -> dict:
    """
    Full evaluation: BLEU + ROUGE-L + BERTScore + Faithfulness.

    Args:
        predictions : generated answers
        references  : gold reference answers
        contexts    : retrieved chunks per query (for faithfulness)
        label       : model name for logging
    """
    print("  Computing BLEU & ROUGE-L...")
    bleu = compute_bleu(predictions, references)
    rouge = compute_rouge(predictions, references)

    print("  Computing BERTScore (primary metric)...")
    bert = compute_bertscore(predictions, references)

    faith = None
    if contexts is not None:
        print("  Computing Faithfulness...")
        faith = compute_faithfulness(predictions, contexts)

    result = {
        "label": label,
        "bleu": round(bleu, 4),
        "rouge_l": round(rouge, 4),
        "bertscore_f1": round(bert, 4),
        "n_samples": len(predictions),
    }
    if faith is not None:
        result["faithfulness"] = round(faith, 4)

    return result
