"""
Run RAG evaluation — v3 abstractive pipeline on true holdout.

Usage:
    set PYTHONIOENCODING=utf-8

    # Quick 10-query sanity check (~1-2 min with Groq API)
    python scripts/run_evaluation.py --quick

    # Full 200-query evaluation (~10-15 min with Groq API)
    python scripts/run_evaluation.py
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Fix encoding for Windows terminals
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag.pipeline import build_rag_pipeline  # noqa: E402
from src.evaluation.metrics import (            # noqa: E402
    compute_improvement, evaluate_pair, evaluate_full,
)
from openai import OpenAI  # noqa: E402

HOLDOUT_PATH = "data/processed/eval_holdout.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick 10-query sanity check",
    )
    parser.add_argument(
        "--sample", type=int, default=200,
        help="Number of eval queries",
    )
    args = parser.parse_args()

    n_queries = min(args.sample, 10) if args.quick else args.sample

    # 1. Load holdout
    print(f"[LOAD] Loading holdout from {HOLDOUT_PATH}...")
    df = pd.read_csv(HOLDOUT_PATH)
    df_sample = df.sample(n=n_queries, random_state=42).reset_index(drop=True)
    questions = df_sample["question"].tolist()
    references = df_sample["answer"].tolist()
    print(f"[LOAD] {len(questions)} evaluation queries loaded")

    # 2. Print a sample to verify
    print()
    print(f"[SAMPLE] Question:  {questions[0][:100]}...")
    print(f"[SAMPLE] Reference: {references[0][:100]}...")

    # 3. Load pipeline
    print()
    print("[RAG] Loading pipeline...")
    pipeline = build_rag_pipeline()
    print("[RAG] Pipeline ready")

    # 4. Run RAG pipeline
    print()
    print(f"[RAG] Generating answers for {len(questions)} queries...")
    rag_outputs = []
    rag_latencies = []
    rag_contexts = []

    for i, q in enumerate(questions):
        start = time.time()
        result = pipeline.answer(q)
        elapsed = (time.time() - start) * 1000

        rag_outputs.append(result["answer_raw"])
        rag_latencies.append(elapsed)

        # Extract contexts from the answer result to avoid a second retrieval call
        # (pipeline.answer already retrieves internally)
        contexts = [
            s.get("context", "") + " " + s.get("answer", "")
            for s in result.get("retrieved_sources", [])
        ]
        # If no sources returned (routing guard returned False), fall back
        if not contexts:
            contexts = [""]
        rag_contexts.append(contexts)

        if (i + 1) % 5 == 0:
            avg = np.mean(rag_latencies)
            print(f"  [{i+1}/{len(questions)}] avg latency: {avg:.0f}ms")

    mean_latency = np.mean(rag_latencies)
    print(f"[RAG] Done. Mean latency: {mean_latency:.0f}ms")

    # Show a sample RAG output
    print()
    print(f"[SAMPLE] RAG output: {rag_outputs[0][:200]}...")

    # 5. Run plain LLM baseline
    groq_key = os.getenv("GROQ_API_KEY")
    groq_client = OpenAI(
        api_key=groq_key,
        base_url="https://api.groq.com/openai/v1",
    )
    groq_model = "meta-llama/llama-4-scout-17b-16e-instruct"

    print()
    print(f"[LLM] Generating plain answers for {len(questions)} queries...")
    llm_outputs = []
    for i, q in enumerate(questions):
        response = groq_client.chat.completions.create(
            model=groq_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a medical research assistant. "
                        "Answer medical questions concisely and accurately."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Answer this medical question: {q}",
                },
            ],
            max_tokens=256,
            temperature=0.1,
        )
        output = response.choices[0].message.content.strip()
        llm_outputs.append(output)
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(questions)}]")
    print("[LLM] Done")

    # 6. Compute metrics
    print()
    print("[METRICS] Computing...")
    rag_metrics = evaluate_full(
        rag_outputs, references, contexts=rag_contexts, label="RAG",
    )
    llm_metrics = evaluate_pair(llm_outputs, references, label="Plain LLM")

    bleu_improvement = compute_improvement(
        llm_metrics["bleu"], rag_metrics["bleu"],
    )
    bertscore_rag = rag_metrics.get("bertscore_f1", 0)
    faithfulness = rag_metrics.get("faithfulness", 0)

    # 7. Print results
    print()
    print("=" * 70)
    print("EVALUATION RESULTS - v3 Abstractive RAG Pipeline")
    print("=" * 70)
    header = f"{'Metric':<25} {'RAG':>10} {'Plain LLM':>12} {'Improvement':>14}"
    print(header)
    print("-" * 70)
    print(
        f"{'BLEU':<25} {rag_metrics['bleu']:>10.4f} "
        f"{llm_metrics['bleu']:>12.4f} {bleu_improvement:>13.1f}%"
    )
    rouge_diff = rag_metrics["rouge_l"] - llm_metrics["rouge_l"]
    print(
        f"{'ROUGE-L':<25} {rag_metrics['rouge_l']:>10.4f} "
        f"{llm_metrics['rouge_l']:>12.4f} {rouge_diff:>13.1%}"
    )
    print(
        f"{'BERTScore F1':<25} {bertscore_rag:>10.4f} "
        f"{'---':>12} {'---':>14}"
    )
    print(
        f"{'Faithfulness':<25} {faithfulness:>10.1%} "
        f"{'---':>12} {'---':>14}"
    )
    print(
        f"{'Latency (mean)':<25} {mean_latency:>10.0f}ms "
        f"{'---':>12} {'---':>14}"
    )

    print()
    print("[KPI] Abstractive RAG Targets (true holdout):")
    rl_status = "PASS" if rag_metrics["rouge_l"] >= 0.15 else "FAIL"
    print(
        f"   ROUGE-L >= 0.15:         {rl_status}"
        f"  ({rag_metrics['rouge_l']:.4f})"
    )
    bs_status = "PASS" if bertscore_rag >= 0.80 else "FAIL"
    print(
        f"   BERTScore F1 >= 0.80:     {bs_status}"
        f"  ({bertscore_rag:.4f})"
    )
    bi_status = "PASS" if bleu_improvement >= 6.0 else "FAIL"
    print(
        f"   BLEU improv. >= +6%:      {bi_status}"
        f"  ({bleu_improvement:.1f}%)"
    )
    ft_status = "PASS" if faithfulness >= 0.70 else "FAIL"
    print(
        f"   Faithfulness >= 70%:      {ft_status}"
        f"  ({faithfulness:.1%})"
    )
    lt_status = "PASS" if mean_latency <= 5000 else "FAIL"
    print(
        f"   Latency <= 5000ms:        {lt_status}"
        f"  ({mean_latency:.0f}ms)"
    )

    # 8. Save results
    results_df = pd.DataFrame({
        "question": questions,
        "reference": references,
        "rag_answer": rag_outputs,
        "llm_answer": llm_outputs,
    })
    results_df.to_csv("reports/rag_evaluation_results.csv", index=False)
    print()
    print("[OK] Saved reports/rag_evaluation_results.csv")

    metrics = {
        "n_queries": len(questions),
        "pipeline": "v3_abstractive",
        "holdout_type": "true_generalization",
        "rag": {
            "bleu": rag_metrics["bleu"],
            "rouge_l": rag_metrics["rouge_l"],
            "bertscore_f1": bertscore_rag,
            "faithfulness": faithfulness,
        },
        "llm": {
            "bleu": llm_metrics["bleu"],
            "rouge_l": llm_metrics["rouge_l"],
        },
        "bleu_improvement_pct": round(bleu_improvement, 1),
        "mean_latency_ms": round(mean_latency, 0),
    }
    with open("reports/evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("[OK] Saved reports/evaluation_metrics.json")


if __name__ == "__main__":
    main()
