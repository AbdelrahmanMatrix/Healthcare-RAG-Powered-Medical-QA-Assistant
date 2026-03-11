from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from typing import List, Dict

class MedicalEvaluator:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction().method1

    def calculate_bleu(self, reference: str, candidate: str) -> float:
        ref_tokens = reference.split()
        cand_tokens = candidate.split()
        return sentence_bleu([ref_tokens], cand_tokens, smoothing_function=self.smoothing)

    def calculate_rouge_l(self, reference: str, candidate: str) -> float:
        scores = self.rouge_scorer.score(reference, candidate)
        return scores['rougeL'].fmeasure

    def evaluate_batch(self, references: List[str], candidates: List[str]) -> Dict[str, float]:
        if len(references) != len(candidates):
            raise ValueError("References and candidates must have the same length")

        bleu_scores = [self.calculate_bleu(r, c) for r, c in zip(references, candidates)]
        rouge_scores = [self.calculate_rouge_l(r, c) for r, c in zip(references, candidates)]

        return {
            "avg_bleu": sum(bleu_scores) / len(bleu_scores),
            "avg_rouge_l": sum(rouge_scores) / len(rouge_scores)
        }

if __name__ == "__main__":
    evaluator = MedicalEvaluator()
    ref = "Aspirin is used to treat pain and fever."
    cand = "Aspirin can be used for treating pain and also fever."

    bleu = evaluator.calculate_bleu(ref, cand)
    rouge = evaluator.calculate_rouge_l(ref, cand)

    print(f"BLEU: {bleu:.4f}")
    print(f"ROUGE-L: {rouge:.4f}")
