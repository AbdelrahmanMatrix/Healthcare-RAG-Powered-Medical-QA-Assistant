# -*- coding: utf-8 -*-
"""Final-validation retrieval diagnostics (read-only measurements).

Runs the REPAIRED retrieval pipeline over the exact NB08 evaluation sample
(200 queries, seed 42, from data/processed/eval_holdout.csv) and measures
retrieval-only behavior. No generation, no score changes, no optimization.

Per query, mirrors production routing (src/pipeline.py + retrieve_by_category):
  1. classifier category + per-class softmax (all_scores)
  2. FAISS raw pool (top_k x per-category expansion factor), cosine inner product
  3. BM25 raw hits, threshold survivors (threshold = settings.BM25_THRESHOLD)
  4. fusion: BM25 survivors prepended, dedup by chunk_id (keep-first)
  5. full merged pool sorted by (category_score DESC, faiss_score DESC), truncate top_k
  6. cross-encoder rerank (final authority) + category quality-guard check

Outputs:
  reports/retrieval_diagnostics.json
  reports/retrieval_diagnostics.md
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path

# Windows/torch quirk: torch MUST be imported before pandas/numpy/transformers,
# otherwise c10.dll initialization fails (WinError 1114) on this machine.
import torch  # noqa: F401  (import-order side effect)

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

N_EVAL = 200
SEED = 42
TOP_K = 30
HOLDOUT_PATH = PROJECT_ROOT / "data" / "processed" / "eval_holdout.csv"
OUT_JSON = PROJECT_ROOT / "reports" / "retrieval_diagnostics.json"
OUT_MD = PROJECT_ROOT / "reports" / "retrieval_diagnostics.md"
CHECKPOINT = PROJECT_ROOT / "reports" / ".rdiag_checkpoint.jsonl"

print("Loading holdout ...")
df = pd.read_csv(HOLDOUT_PATH)
sample = df.sample(n=min(N_EVAL, len(df)), random_state=SEED).reset_index(drop=True)
questions = sample["question"].tolist()
print(f"Holdout rows: {len(df):,} | eval sample: {len(questions)} (seed={SEED})")

print("Building repaired RAG pipeline (components load lazily) ...")
t0 = time.time()
from src.rag.pipeline import build_rag_pipeline, _truncate_words  # noqa: E402
pipeline = build_rag_pipeline()
print(f"Pipeline ready in {time.time() - t0:.1f}s")
print(f"  index.ntotal   = {pipeline.index.ntotal:,}")
print(f"  use_bm25       = {pipeline._use_bm25}")
print(f"  use_reranker   = {pipeline._use_reranker}")
print(f"  use_classifier = {getattr(pipeline, '_use_classifier', False)}")
print(f"  bm25_threshold = {pipeline._bm25_threshold}")
print(f"  top_k          = {pipeline.top_k}")

# Holdout exclusion sanity check (only possible if the CSV carries chunk ids)
holdout_ids = set(sample["chunk_id"].astype(int)) if "chunk_id" in sample.columns else None
if holdout_ids is not None:
    overlap = holdout_ids.intersection(range(pipeline.index.ntotal))
    print(f"LEAK CHECK: holdout chunk_ids inside index range: {len(overlap)} (must be 0)")
else:
    overlap = None
    print("LEAK CHECK: no chunk_id column in holdout CSV - id-level check not possible here")

enc = pipeline.encoder
bm25 = pipeline.bm25
reranker = pipeline.reranker
index = pipeline.index
mapping = pipeline.mapping_df
expansion = pipeline._category_expansion
threshold = pipeline._bm25_threshold


def faiss_raw(query, search_k):
    v = enc.encode([query], convert_to_numpy=True).astype(np.float32)
    import faiss
    faiss.normalize_L2(v)
    D, I = index.search(v, search_k)
    return [(int(I[0, r]), float(D[0, r])) for r in range(search_k) if int(I[0, r]) >= 0]


def dedup_keep_first(pairs):
    seen, out = set(), []
    for item in pairs:
        cid = item[0]
        if cid not in seen:
            seen.add(cid)
            out.append(item)
    return out


def rerank_text(cid):
    row = mapping.iloc[cid]
    return row["answer"] + " " + _truncate_words(row["context"], 100)


rows = []
t_start = time.time()
skipped_classifier = 0
skipped_needs_rag = 0

# checkpoint/resume: reuse completed per-query records across invocations
# (this machine caps command runtime, and model+BM25 load alone takes minutes)
FINGERPRINT = f"{HOLDOUT_PATH.stat().st_size}|{HOLDOUT_PATH.stat().st_mtime_ns}|{len(df)}|{SEED}|{N_EVAL}"
done = {}
if CHECKPOINT.exists():
    fp_file = CHECKPOINT.with_suffix(".fp")
    if fp_file.exists() and fp_file.read_text(encoding="utf-8").strip() == FINGERPRINT:
        for line in CHECKPOINT.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["n"]] = r
        print(f"Resuming: {len(done)}/{len(questions)} queries already checkpointed")
    else:
        print("Checkpoint fingerprint mismatch - starting fresh")
        CHECKPOINT.unlink()
CHECKPOINT.with_suffix(".fp").write_text(FINGERPRINT, encoding="utf-8")
ckpt = CHECKPOINT.open("a", encoding="utf-8")


def classify_rec(n, rec):
    if rec.get("skipped_guard"):
        return "guard"
    if "classifier_error" in rec:
        return "error"
    return "valid"


for n, q in enumerate(questions):
    if n in done:
        rec = done[n]
        kind = classify_rec(n, rec)
        rows.append(rec)
        if kind == "guard":
            skipped_needs_rag += 1
        elif kind == "error":
            skipped_classifier += 1
        continue

    rec = {"n": n}

    if not pipeline._needs_retrieval(q):
        skipped_needs_rag += 1
        rec["skipped_guard"] = True
        rows.append(rec)
        continue
    rec["skipped_guard"] = False

    try:
        cls = pipeline._classifier.predict_with_confidence(q)
    except Exception as e:  # noqa: BLE001
        skipped_classifier += 1
        rec["classifier_error"] = str(e)[:120]
        rows.append(rec)
        continue
    pred, conf, all_scores = cls["category"], cls["confidence"], cls["all_scores"]
    rec.update(pred_category=pred, confidence=round(conf, 4))

    expanded = f"{pred.lower()}: {q}" if pred and pred != "General" else q

    # 2. FAISS raw pool (expanded depth, as in retrieve_by_category)
    factor = expansion.get(pred, 3)
    search_k = min(TOP_K * factor, index.ntotal)
    faiss_pairs = faiss_raw(expanded, search_k)
    faiss_top30_ids = [cid for cid, _ in faiss_pairs[:TOP_K]]

    # 3. BM25 raw + threshold survivors
    bm25_hits = bm25.retrieve(q, top_k=TOP_K)
    bm25_survivors = [(int(h["chunk_id"]), float(h["bm25_score"])) for h in bm25_hits
                      if float(h["bm25_score"]) > threshold]

    # 4. fusion exactly as production: BM25 survivors prepended, keep-first dedup
    merged = dedup_keep_first(bm25_survivors + faiss_pairs)
    n_unique_after_fusion = len(merged)

    # 5. production order: score FULL merged pool, sort by (cat, faiss), truncate k
    # (BM25-only candidates get faiss_score 0.0 — same as production, which only
    # carries faiss_score on FAISS-sourced rows)
    faiss_score_map = dict(faiss_pairs)
    survivor_ids = [cid for cid, _ in bm25_survivors]
    scored = [
        {"chunk_id": cid,
         "faiss_score": faiss_score_map.get(cid, 0.0),
         "category_score": all_scores.get(mapping.iloc[cid].get("category", "Unknown"), 0.0)}
        for cid, _ in merged
    ]
    pool = sorted(scored, key=lambda x: (x["category_score"], x["faiss_score"]), reverse=True)[:TOP_K]

    # 6. rerank once per unique candidate (pool + FAISS top-30), reuse scores
    cat_pool_ids = [c["chunk_id"] for c in pool]
    unique_ids = list(dict.fromkeys(cat_pool_ids + faiss_top30_ids))
    t_q0 = time.time()
    pairs = [[expanded, rerank_text(cid)] for cid in unique_ids]
    r_scores = reranker.predict(pairs)
    rerank_ms = (time.time() - t_q0) * 1000
    score_map = {cid: float(s) for cid, s in zip(unique_ids, r_scores)}

    pool_sorted = sorted(cat_pool_ids, key=lambda c: score_map[c], reverse=True)
    faiss_sorted = sorted(faiss_top30_ids, key=lambda c: score_map[c], reverse=True)
    final_top1 = pool_sorted[0] if pool_sorted else None
    faiss_only_top1 = faiss_sorted[0] if faiss_sorted else None
    top3_scores = [score_map[c] for c in pool_sorted[:3]]
    quality = float(np.mean(top3_scores)) if top3_scores else 0.0

    # contributions
    faiss_ids_set = set(cid for cid, _ in faiss_pairs)
    faiss_top30_set = set(faiss_top30_ids)
    bm25_only_vs_full_faiss = [cid for cid in survivor_ids if cid not in faiss_ids_set]
    bm25_only_vs_top30 = [cid for cid in survivor_ids if cid not in faiss_top30_set]
    faiss_only_contrib = [cid for cid in faiss_ids_set if cid not in set(survivor_ids)]

    rec.update(
        faiss_pool_size=search_k,
        bm25_raw=len(bm25_hits),
        bm25_survivors=len(bm25_survivors),
        unique_after_fusion=n_unique_after_fusion,
        final_pool_size=len(pool),
        bm25_only_not_in_full_faiss=len(bm25_only_vs_full_faiss),
        bm25_only_not_in_faiss_top30=len(bm25_only_vs_top30),
        faiss_only_candidates=len(faiss_only_contrib),
        pred_top1_cat_pool=cat_pool_ids[0] if cat_pool_ids else None,
        final_top1=final_top1,
        faiss_only_top1=faiss_only_top1,
        rerank_top3_scores=[round(s, 4) for s in top3_scores],
        quality=round(quality, 4),
        cat_fallback_triggered=bool(pool) and quality < 1.0,
        rerank_ms=round(rerank_ms, 1),
    )
    rows.append(rec)
    ckpt.write(json.dumps(rec, ensure_ascii=False) + "\n")
    ckpt.flush()
    if (n + 1) % 10 == 0:
        print(f"  {n + 1}/{len(questions)} done | elapsed {time.time() - t_start:.0f}s")

ckpt.close()

# Aggregates
valid = [r for r in rows if not r.get("skipped_guard") and "classifier_error" not in r]
print(f"\nValid measured queries: {len(valid)} "
      f"(guard-skipped: {skipped_needs_rag}, classifier-errors: {skipped_classifier})")


def pct(x, n):
    return round(x / n * 100, 1) if n else 0.0


rerank_flips = sum(1 for r in valid if r["final_top1"] is not None
                   and r["pred_top1_cat_pool"] is not None and r["final_top1"] != r["pred_top1_cat_pool"])
rerank_flips_vs_faiss = sum(1 for r in valid
                            if r["final_top1"] != r["faiss_only_top1"])
cat_fallbacks = sum(1 for r in valid if r["cat_fallback_triggered"])
top1_agree = sum(1 for r in valid if r["final_top1"] == r["faiss_only_top1"])
bm25_adds_any = sum(1 for r in valid if r["bm25_only_not_in_full_faiss"] > 0)
bm25_adds_top30 = sum(1 for r in valid if r["bm25_only_not_in_faiss_top30"] > 0)
faiss_adds_any = sum(1 for r in valid if r["faiss_only_candidates"] > 0)

pool_sizes = [r["unique_after_fusion"] for r in valid]
surv_counts = [r["bm25_survivors"] for r in valid]
rerank_ms_list = [r["rerank_ms"] for r in valid]
quality_list = [r["quality"] for r in valid]
per_category = Counter(r["pred_category"] for r in valid)
conf_values = [r["confidence"] for r in valid]

summary = {
    "metadata": {
        "n_eval_queries": len(questions),
        "seed": SEED,
        "holdout_rows": len(df),
        "index_ntotal": index.ntotal,
        "top_k": TOP_K,
        "bm25_threshold": threshold,
        "category_expansion": dict(expansion),
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "embedding_model": "pritamdeka/S-PubMedBert-MS-MARCO",
        "leak_check_overlap": len(overlap) if overlap is not None else "not_verifiable_no_chunk_id_column",
        "note": "Retrieval-only diagnostics. No generation, no LLM metrics, no optimization.",
    },
    "guard": {
        "skipped_needs_rag": skipped_needs_rag,
        "classifier_errors": skipped_classifier,
        "measured_queries": len(valid),
    },
    "A_top_k_behavior": {
        "unique_candidates_after_fusion": {
            "mean": round(float(np.mean(pool_sizes)), 1),
            "median": float(np.median(pool_sizes)),
            "min": int(min(pool_sizes)),
            "max": int(max(pool_sizes)),
        },
        "queries_where_fusion_pool_exceeded_top_k": sum(1 for s in pool_sizes if s > TOP_K),
        "final_pool_size": TOP_K,
    },
    "B_contributions": {
        "bm25_survivors_per_query": {
            "mean": round(float(np.mean(surv_counts)), 2),
            "median": float(np.median(surv_counts)),
            "max": int(max(surv_counts)),
            "queries_with_0_survivors": sum(1 for s in surv_counts if s == 0),
            "queries_with_at_least_1_survivor": sum(1 for s in surv_counts if s > 0),
        },
        "bm25_adds_candidate_not_in_full_faiss_pool": {
            "count": bm25_adds_any, "rate_pct": pct(bm25_adds_any, len(valid)),
        },
        "bm25_adds_candidate_not_in_faiss_top30": {
            "count": bm25_adds_top30, "rate_pct": pct(bm25_adds_top30, len(valid)),
        },
        "faiss_adds_candidate_not_in_bm25_survivors": {
            "count": faiss_adds_any, "rate_pct": pct(faiss_adds_any, len(valid)),
        },
        "rerank_latency_ms_per_predict_call": {
            "mean": round(float(np.mean(rerank_ms_list)), 1),
            "p50": round(float(np.percentile(rerank_ms_list, 50)), 1),
            "p95": round(float(np.percentile(rerank_ms_list, 95)), 1),
            "device": "CPU",
        },
        "top3_reranker_quality": {
            "mean": round(float(np.mean(quality_list)), 4),
            "min": round(float(min(quality_list)), 4),
            "max": round(float(max(quality_list)), 4),
        },
    },
    "C_behaviors": {
        "reranker_changes_top_result_vs_category_pool": {
            "count": rerank_flips, "rate_pct": pct(rerank_flips, len(valid)),
        },
        "final_top1_differs_from_faiss_only_top1": {
            "count": rerank_flips_vs_faiss, "rate_pct": pct(rerank_flips_vs_faiss, len(valid)),
        },
        "final_top1_agrees_with_faiss_only_top1_pct": pct(top1_agree, len(valid)),
        "category_quality_guard_fallback_would_trigger": {
            "count": cat_fallbacks, "rate_pct": pct(cat_fallbacks, len(valid)),
        },
    },
    "D_categories": {
        "predicted_category_counts": dict(per_category),
        "confidence": {
            "mean": round(float(np.mean(conf_values)), 4),
            "min": round(float(min(conf_values)), 4),
        },
    },
    "per_query": rows,
}

OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

md = [
    "# Retrieval-Only Diagnostics (Final Validation)",
    "",
    f"**Eval sample:** {len(questions)} queries, seed {SEED}, from `data/processed/eval_holdout.csv` (2,000-row RAG holdout)",
    f"**Index rows:** {index.ntotal:,} | **top_k:** {TOP_K} | **BM25 threshold:** {threshold} | **expansion:** {dict(expansion)}",
    f"**Leak check:** holdout chunk_ids present in index range: {len(overlap) if overlap is not None else 'not verifiable at id level (no chunk_id column in CSV)'}",
    f"**Measured queries:** {len(valid)} of {len(questions)} (guard-skipped: {skipped_needs_rag}, classifier errors: {skipped_classifier})",
    "",
    "## A. Top-k / fusion behavior",
    f"- Unique candidates after fusion (before truncate-to-{TOP_K}): mean {summary['A_top_k_behavior']['unique_candidates_after_fusion']['mean']}, median {summary['A_top_k_behavior']['unique_candidates_after_fusion']['median']}, range {summary['A_top_k_behavior']['unique_candidates_after_fusion']['min']}-{summary['A_top_k_behavior']['unique_candidates_after_fusion']['max']}",
    f"- Queries where the fusion pool exceeded top_k (truncation applied): {summary['A_top_k_behavior']['queries_where_fusion_pool_exceeded_top_k']}/{len(valid)}",
    "",
    "## B. Retrieval contributions",
    f"- BM25 threshold survivors per query: mean {summary['B_contributions']['bm25_survivors_per_query']['mean']}, median {summary['B_contributions']['bm25_survivors_per_query']['median']}, max {summary['B_contributions']['bm25_survivors_per_query']['max']}",
    f"- Queries with at least 1 BM25 survivor entering the pool: {summary['B_contributions']['bm25_survivors_per_query']['queries_with_at_least_1_survivor']}/{len(valid)}",
    f"- BM25 added a candidate absent from the FULL expanded FAISS pool: {bm25_adds_any}/{len(valid)} ({summary['B_contributions']['bm25_adds_candidate_not_in_full_faiss_pool']['rate_pct']}%)",
    f"- BM25 added a candidate absent from FAISS top-30: {bm25_adds_top30}/{len(valid)} ({summary['B_contributions']['bm25_adds_candidate_not_in_faiss_top30']['rate_pct']}%)",
    f"- FAISS contributed candidates absent from BM25 survivors: {faiss_adds_any}/{len(valid)} ({summary['B_contributions']['faiss_adds_candidate_not_in_bm25_survivors']['rate_pct']}%)",
    f"- Cross-encoder rerank latency per predict call ({TOP_K} pairs): mean {summary['B_contributions']['rerank_latency_ms_per_predict_call']['mean']} ms, p95 {summary['B_contributions']['rerank_latency_ms_per_predict_call']['p95']} ms (CPU)",
    "",
    "## C. Reranker / guard behavior",
    f"- Reranker changed the top result vs the category-priority pool order: {rerank_flips}/{len(valid)} ({summary['C_behaviors']['reranker_changes_top_result_vs_category_pool']['rate_pct']}%)",
    f"- Final top-1 differs from FAISS-only top-1: {rerank_flips_vs_faiss}/{len(valid)} ({summary['C_behaviors']['final_top1_differs_from_faiss_only_top1']['rate_pct']}%)",
    f"- Final top-1 agrees with FAISS-only top-1: {summary['C_behaviors']['final_top1_agrees_with_faiss_only_top1_pct']}%",
    f"- Category quality-guard fallback (top-3 mean reranker score < 1.0) would trigger: {cat_fallbacks}/{len(valid)} ({summary['C_behaviors']['category_quality_guard_fallback_would_trigger']['rate_pct']}%)",
    "",
    "## D. Classifier routing (retrieval context)",
    f"- Predicted categories: {dict(per_category)}",
    f"- Confidence: mean {summary['D_categories']['confidence']['mean']}, min {summary['D_categories']['confidence']['min']}",
    f"- Guard-skipped (no-retrieval) queries: {skipped_needs_rag}",
    "",
    "**Status:** measurements only. No optimization performed. Generation-dependent metrics NOT run here "
    "(canonical LLM evaluation is blocked on this machine: no GROQ_API_KEY).",
]
OUT_MD.write_text("\n".join(md), encoding="utf-8")
print(f"\nSaved: {OUT_JSON}")
print(f"Saved: {OUT_MD}")
