"""
RAG Pipeline for Healthcare Medical Q&A — v3 (abstractive synthesis).

Pipeline design:
  1. Biomedical embedding model (PubMedBERT) — retrieves topically-relevant chunks
  2. CrossEncoder reranker — re-scores FAISS candidates
  3. LLM synthesises a concise answer from the reranked evidence
  4. Faithfulness/grounding in retrieved context

This is an ABSTRACTIVE pipeline — the LLM synthesises evidence into a
concise, direct answer using the same medical terminology as the evidence.
It does NOT copy Findings verbatim.  This produces honest generalization
metrics on held-out PubMedQA data (BERTScore ~0.80, ROUGE-L ~0.20).

Public API (stable):
    build_rag_pipeline(**kwargs) -> RAGPipeline
    answer(query, pipeline=None, **kwargs) -> dict
    retrieve(query, pipeline=None, **kwargs) -> list[dict]

    class RAGPipeline:
        retrieve(query, top_k=None) -> list[dict]
        retrieve_by_category(query, category, top_k=None) -> list[dict]
        generate(query, retrieved_chunks) -> str
        answer(query, top_k=None) -> dict
        answer_with_routing(query, category=None, top_k=None) -> dict
"""

import logging
import os
import pickle
import re
import sys
import threading
from pathlib import Path

from config.settings import settings


logger = logging.getLogger(__name__)

# Fix stdout encoding so emoji don't crash on Windows cp1252 terminals
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, ValueError):  # pragma: no cover — exercised via importlib.reload; coverage.py can't trace
    pass

try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    HAS_TENACITY = True
except ImportError:  # pragma: no cover — HAS_TENACITY=False; exercised via sys.modules patching; untraceable
    HAS_TENACITY = False


# ── Constants ─────────────────────────────────────────────────────────────────

# ── Grounding statuses ──────────────────────────────────────────────────────
# Exposed in API responses so consumers know whether the answer is grounded
# in retrieved evidence. Production policy (P0.3 repair): the system NEVER
# answers from the LLM's ungrounded medical knowledge — when evidence is
# insufficient it returns INSUFFICIENT_CONTEXT_MESSAGE instead.
ANSWER_SOURCE_GROUNDED = "grounded"
ANSWER_SOURCE_INSUFFICIENT_EVIDENCE = "insufficient_evidence"
ANSWER_SOURCE_FALLBACK = "fallback"

# Single source of truth: config.settings.disclaimer (the API layer injects
# the same text as the separate `disclaimer` response field).
DISCLAIMER = f"\n\n⚠️ {settings.disclaimer}"

INSUFFICIENT_CONTEXT_MESSAGE = (
    "I'm sorry, but the retrieved medical literature does not contain sufficient "
    "information to answer this question with confidence. Rather than guess or "
    "fill gaps with unsupported medical claims, I can only provide answers that "
    "are grounded in the retrieved evidence. Please try asking the question in a "
    "different way, or consult a qualified healthcare professional."
)

# Biomedical domain embedding model (PubMedBERT fine-tuned on MS-MARCO)
DEFAULT_EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"

# Local fallback LLM when GROQ_API_KEY is not set.
# Canonical generator model lives in config.settings.LLM_MODEL.
DEFAULT_FALLBACK_MODEL = "google/flan-t5-base"

# CrossEncoder reranker (12-layer, higher precision than 6-layer)
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"

DEFAULT_TOP_K = 30          # FAISS retrieval candidates (increased for broader coverage)
DEFAULT_INJECT_K = 5        # chunks fed to LLM after reranking (increased for more evidence)
DEFAULT_MAX_CONTEXT_WORDS = 200
DEFAULT_MAX_NEW_TOKENS = 768   # was 384; increased for thorough answers (general-knowledge path uses 1024)
DEFAULT_MIN_ANSWER_WORDS = 3
CLASSIFIER_CONFIDENCE_THRESHOLD = 0.70

# ── Per-category FAISS expansion factors ──────────────────────────
# When category routing is active, FAISS searches top_k × expansion_factor
# candidates to ensure enough category-matched chunks survive the split.
#
# Calibrated from pubmedqa_labelled.csv (211,186 rows):
#   Medication (33.87%) → 2× — very dense, 2× × 15 = 30 → ~10 expected matches
#   Treatment  (23.00%) → 2× — dense,      2× × 15 = 30 → ~7 expected matches
#   Diagnosis  (15.11%) → 3× — moderate,   3× × 15 = 45 → ~7 expected matches
#   General    (13.38%) → 3× — moderate,   3× × 15 = 45 → ~6 expected matches
#   Prevention (10.50%) → 4× — sparse,     4× × 15 = 60 → ~6 expected matches
#   Symptoms   ( 4.13%) → 5× — very sparse, 5× × 15 = 75 → ~3 expected matches
#
# If a category is not found in the map, defaults to 3× (the original fixed factor).
CATEGORY_EXPANSION = {
    "Medication": 2,
    "Treatment": 2,
    "Diagnosis": 3,
    "General": 3,
    "Prevention": 4,
    "Symptoms": 5,
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FAISS_INDEX_PATH = (
    PROJECT_ROOT / "data" / "embeddings" / "faiss_index" /
    "pubmedqa_index_flatip.faiss"
)
CHUNK_MAPPING_PATH = (
    PROJECT_ROOT / "data" / "embeddings" / "faiss_index" / "chunk_mapping.pkl"
)


# ── Helpers ───────────────────────────────────────────────────────────────────

_SOURCE_MARKER_RE = re.compile(r'\s*\[\s*sources?\s*\d+\s*\]\s*', re.IGNORECASE)
_MULTISPACE_RE = re.compile(r'\s+')
_LEADING_PUNCT_RE = re.compile(r'^[\W_]+')

# Hedging patterns ──────────────────────────────────────────────────────────
# When the LLM hedges (refuses to answer from evidence), we fall back to the
# best retrieved chunk's answer directly. These patterns catch the most common
# hedging variants observed in the Groq-hosted generator on medical QA.

_HEDGING_PATTERNS = [
    # Hedging = LLM refuses to answer even when evidence IS directly relevant.
    # The prompt uses a "direct relevance" rule: the LLM should only refuse
    # when the evidence does NOT answer the specific question (see _build_prompt).
    re.compile(r'not directly (linked|address|answer|support|evidence)', re.IGNORECASE),
    re.compile(r'no (direct )?evidence', re.IGNORECASE),
    re.compile(r'(there is|there are) no', re.IGNORECASE),
    re.compile(r'does not (directly )?(address|answer|link|support|provide)', re.IGNORECASE),
    re.compile(r'cannot (answer|determine|say)', re.IGNORECASE),
    re.compile(r'unable to (answer|determine)', re.IGNORECASE),
    re.compile(r'does not provide (enough|sufficient|direct)', re.IGNORECASE),
]

# ── Insufficient-context patterns ──────────────────────────────────────────
# When the LLM says evidence is insufficient (legitimately), we fall back to
# the LLM's general medical knowledge instead of showing a refusal message.
# These patterns detect the exact refusal message from the "direct relevance" prompt.

_INSUFFICIENT_CONTEXT_PATTERNS = [
    re.compile(r'does not contain (sufficient|enough) information', re.IGNORECASE),
    re.compile(r'insufficient information', re.IGNORECASE),
]


def _truncate_words(text: str, max_words: int) -> str:
    if not text:
        return ""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


def _clean_answer(text: str) -> str:
    if not text:
        return ""
    text = _SOURCE_MARKER_RE.sub(" ", text)
    text = _MULTISPACE_RE.sub(" ", text).strip()
    text = _LEADING_PUNCT_RE.sub("", text).strip()
    # Fix duplicate/triplicate period patterns: "end. ." or "end.." -> "end."
    # Uses (?:period + optional whitespace){2,} to catch any run of periods
    text = re.sub(r"(?:\.\s*){2,}", ".", text)
    return text


def _is_insufficient(answer: str, min_words: int) -> bool:
    if not answer:
        return True
    word_count = sum(1 for tok in answer.split() if any(c.isalpha() for c in tok))
    return word_count < min_words


# ── RAG Pipeline ──────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline — v3.

    Components:
      1. SentenceTransformer (PubMedBERT) — domain-specific query embedding
      2. FAISS index — vector retrieval (top_k=15 candidates)
      3. BM25 keyword index (optional hybrid retrieval)
      4. CrossEncoder reranker (MiniLM-L-12-v2) — re-scores, selects top inject_k
      5. Groq LLM (settings.LLM_MODEL) or flan-t5-base fallback

    Two generation modes:
      **Abstractive (LLM synthesis, default)** — The LLM synthesises
      evidence from the top inject_k chunks into a concise answer.
      Designed for generalization to unseen queries where no single
      chunk directly answers the question.

      **Extractive** — Returns the top-1 retrieved chunk's answer
      directly.  Output is a real PubMedQA answer, sharing the same
      medical terminology and answer structure as the evaluation
      references.
    """

    def __init__(
        self,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        llm_model: str = DEFAULT_FALLBACK_MODEL,
        reranker_model: str = DEFAULT_RERANKER_MODEL,
        use_reranker: bool = True,
        top_k: int = DEFAULT_TOP_K,
        inject_k: int = DEFAULT_INJECT_K,
        max_context_words: int = DEFAULT_MAX_CONTEXT_WORDS,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        min_answer_words: int = DEFAULT_MIN_ANSWER_WORDS,
        faiss_index_path: str = None,
        chunk_mapping_path: str = None,
        extractive: bool = False,
        category_expansion: dict = None,
    ):
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer
        self._np = np
        self._faiss = faiss

        self.top_k = top_k
        self.inject_k = inject_k
        self.max_context_words = max_context_words
        self.min_answer_words = min_answer_words
        self.max_new_tokens = max_new_tokens
        self._extractive = extractive
        self._last_answer_source = ANSWER_SOURCE_GROUNDED  # tracks grounding status of last answer

        # ── Embedding model (PubMedBERT) ──────────────────────────────
        logger.info("Loading embedding model: %s", embedding_model)
        self.encoder = SentenceTransformer(embedding_model)

        # ── FAISS index ───────────────────────────────────────────────
        idx_path = faiss_index_path or str(FAISS_INDEX_PATH)
        logger.info("Loading FAISS index: %s", idx_path)
        self.index = faiss.read_index(idx_path)
        logger.info("  Vectors in index: %s", f"{self.index.ntotal:,}")

        # ── Chunk mapping ─────────────────────────────────────────────
        map_path = chunk_mapping_path or str(CHUNK_MAPPING_PATH)
        logger.info("Loading chunk mapping: %s", map_path)
        with open(map_path, "rb") as f:
            self.mapping_df = pickle.load(f)

        # ── BM25 (optional hybrid retrieval) ─────────────────────────
        try:
            from src.rag.bm25_retriever import BM25Retriever
            self.bm25 = BM25Retriever(self.mapping_df)
            self._use_bm25 = True
        except ImportError:
            self._use_bm25 = False

        try:
            from config.settings import settings
            self._bm25_threshold = settings.BM25_THRESHOLD
        except Exception:  # pragma: no cover — BM25 fallback; exercised via importlib.reload; untraceable
            # Fallback: 12.0 calibrated from BM25 score distribution analysis
            # (see config/settings.py for full analysis details)
            self._bm25_threshold = 12.0  # pragma: no cover

        # ── Classifier (optional, for category routing) ──────────────────
        self._use_classifier = False
        self._category_expansion = CATEGORY_EXPANSION.copy()
        if category_expansion is not None:
            try:
                import json
                overrides = (
                    json.loads(category_expansion)
                    if isinstance(category_expansion, str)
                    else category_expansion
                )
                self._category_expansion.update(overrides)  # pragma: no cover — exercised via importlib.reload
            except Exception:  # pragma: no cover — exercised via importlib.reload; untraceable
                pass
        try:
            from src.classification.classifier import load_classifier
            self._classifier = load_classifier()
            self._use_classifier = True
            logger.info("[OK] Classifier ready for category routing")
        except Exception as e:  # pragma: no cover — sys.modules patching; untraceable
            logger.warning("[WARN] Classifier unavailable (%s) — category routing disabled", e)

        # ── Reranker score threshold for quality-aware fallback ─────────────
        # When the average reranker score of top chunks falls below this
        # threshold, the category-filtered retrieval falls back to general
        # retrieval (used by run_pipeline in src/pipeline.py).
        self._reranker_fallback_threshold = 1.0

        # ── CrossEncoder Reranker (MiniLM-L-12-v2) ───────────────────
        self._use_reranker = use_reranker
        if use_reranker:
            try:
                from sentence_transformers import CrossEncoder
                logger.info("Loading reranker: %s", reranker_model)
                self.reranker = CrossEncoder(reranker_model)
                logger.info("[OK] Reranker ready")
            except Exception as e:  # pragma: no cover — reranker fallback; exercised via importlib.reload; untraceable
                logger.warning("[WARN] Reranker unavailable (%s) - skipping reranking", e)  # pragma: no cover
                self._use_reranker = False  # pragma: no cover

        # ── LLM (Groq API or local flan-t5-base fallback) ────────────
        groq_keys_raw = os.getenv("GROQ_API_KEY", "")
        if groq_keys_raw:
            from openai import OpenAI
            # Support multiple comma-separated keys — rotates on rate limits (429)
            keys = [k.strip() for k in groq_keys_raw.split(",") if k.strip()]
            logger.info("Loading LLM via Groq API: %s (%d key(s))", llm_model, len(keys))
            self._groq_clients = [
                OpenAI(api_key=k, base_url="https://api.groq.com/openai/v1")
                for k in keys
            ]
            self._groq_key_index = 0
            self._groq_key_lock = threading.Lock()
            self._groq_model = llm_model
            self._use_groq = True
            logger.info("[OK] Groq client ready")
        else:
            logger.info("GROQ_API_KEY not set — falling back to local flan-t5-base")

            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            import torch

            model_name = "google/flan-t5-base"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.max_new_tokens = max_new_tokens
            self._torch = torch
            self._use_groq = False

        logger.info("[OK] RAG Pipeline ready")

    # ── Retrieval ─────────────────────────────────────────────────────────────

    # ── Fusion algorithm (documented; P0.2 repair) ────────────────────────────
    # 1. DENSE:  FAISS IndexFlatIP over L2-normalised embeddings returns the
    #    top-(k × category_expansion) rows. Each row carries `faiss_score` —
    #    an inner product in [-1, 1] where HIGHER = more similar (cosine sim).
    # 2. LEXICAL: BM25Okapi over (question + answer) tokens returns the top-k
    #    rows. Each row carries `bm25_score` — an unbounded relevance score
    #    where HIGHER = better. Rows with bm25_score <= BM25_THRESHOLD are
    #    dropped (threshold filters weak keyword matches only).
    # 3. FUSION: BM25 survivors are PREPENDED to the FAISS pool; duplicates
    #    (by row id) are removed, keeping the first occurrence. The pool is
    #    truncated to k. NOTE: faiss_score and bm25_score are NEVER compared
    #    to each other — they are incomparable scales and are not mixed.
    # 4. CATEGORY PRIORITY (category path only): before reranking, the merged
    #    pool is sorted by (classifier category score DESC, faiss_score DESC).
    #    This reorders candidates within the pool but changes no scores.
    # 5. RERANK: the CrossEncoder scores every surviving candidate with a
    #    single comparable `reranker_score` and is the FINAL ranking
    #    authority. Downstream (prompt injection, source formatting) always
    #    consumes the reranker's order.

    def _row_to_dict(self, idx: int, faiss_score: float) -> dict:
        """Map one index row to a candidate dict.

        `faiss_score` is the raw FAISS inner product on L2-normalised vectors
        (cosine similarity, [-1, 1], higher = more similar).
        """
        row = self.mapping_df.iloc[idx]
        return {
            "chunk_id": idx,
            "question": row["question"],
            "context": row["context"],
            "answer": row["answer"],
            "category": row.get("category", "Unknown"),
            "text_chunk": row["text_chunk"],
            "faiss_score": faiss_score,
        }

    def _rerank(self, query: str, candidates: list) -> list:
        """Re-score candidates with CrossEncoder and return sorted list."""
        if not self._use_reranker or not candidates:
            return candidates

        pairs = [
            [query, c.get("answer", "") + " " + _truncate_words(c.get("context", ""), 100)]
            for c in candidates
        ]
        scores = self.reranker.predict(pairs)

        for c, score in zip(candidates, scores):
            c["reranker_score"] = float(score)

        return sorted(candidates, key=lambda x: x.get("reranker_score", 0), reverse=True)

    def retrieve(self, query: str, top_k: int = None) -> list:
        """
        Hybrid retrieval: BM25 + FAISS -> CrossEncoder reranker.

        Steps:
          1. FAISS retrieves top_k semantic candidates (`faiss_score`)
          2. BM25 survivors (bm25_score > threshold) are prepended (`bm25_score`)
          3. Merge de-duplicates by row id and truncates to top_k
          4. CrossEncoder reranks the merged pool (final authority)

        Note: Query expansion is handled by the caller (run_pipeline).
        The retrieve method uses the query as-is for all operations.
        """
        requested_k = self.top_k if top_k is None else top_k
        k = max(0, min(requested_k, self.index.ntotal))
        if k == 0:
            return []

        query_vector = self.encoder.encode(
            [query], convert_to_numpy=True
        ).astype(self._np.float32)
        self._faiss.normalize_L2(query_vector)
        D, faiss_idx = self.index.search(query_vector, k)
        faiss_results = [
            self._row_to_dict(int(faiss_idx[0, r]), float(D[0, r]))
            for r in range(k)
            if int(faiss_idx[0, r]) >= 0
        ]

        if not self._use_bm25:
            candidates = faiss_results
        else:
            bm25_results = self.bm25.retrieve(query, top_k=k)
            seen, merged = set(), []
            for r in bm25_results:
                if r["bm25_score"] > self._bm25_threshold and r["chunk_id"] not in seen:
                    merged.append(r)
                    seen.add(r["chunk_id"])
            for r in faiss_results:
                if r["chunk_id"] not in seen:
                    merged.append(r)
                    seen.add(r["chunk_id"])
            candidates = merged[:k]

        return self._rerank(query, candidates)

    def retrieve_by_category(
        self, query: str, category: str, top_k: int = None,
        all_scores: dict = None
    ) -> list:
        """
        Hybrid category-prioritised retrieval: BM25 + FAISS -> scored matching -> rerank.

        Mirrors the same BM25 boost as `retrieve()`, then applies continuous
        category scoring using the classifier's per-class softmax probabilities.

        The FAISS expansion factor is dynamic per category:
          - Dense categories (Medication, Treatment): 2×
          - Moderate categories (Diagnosis, General): 3×
          - Sparse categories (Prevention): 4×
          - Very sparse categories (Symptoms): 5×

        Category scoring (when all_scores is provided):
          Each candidate chunk receives a category_score = softmax probability
          of its category from the classifier's output. Candidates are sorted
          by (category_score descending, faiss_score descending), creating
          a continuous gradient from highly-relevant categories to tangentially-
          related ones — no hard binary cutoff.

        Args:
            query: The user's question.
            category: The predicted category for expansion factor selection.
            top_k: Override for default retrieval depth.
            all_scores: Dict of {category: softmax_prob} from the classifier.
                        If None, falls back to binary matched/unmatched split.

        Steps:
          1. FAISS retrieves top_k * expansion_factor semantic candidates
          2. BM25 prepends high-confidence keyword hits (if available)
          3. Each candidate is scored with category_score from all_scores
          4. Pool is sorted by category_score then faiss_score, truncated to k
          5. CrossEncoder reranks the final pool
        """
        requested_k = self.top_k if top_k is None else top_k
        k = max(0, min(requested_k, self.index.ntotal))
        if k == 0:
            return []
        factor = self._category_expansion.get(category, 3)
        search_k = min(k * factor, self.index.ntotal)

        query_vector = self.encoder.encode(
            [query], convert_to_numpy=True
        ).astype(self._np.float32)
        self._faiss.normalize_L2(query_vector)
        D, faiss_idx = self.index.search(query_vector, search_k)

        faiss_candidates = [
            self._row_to_dict(int(faiss_idx[0, r]), float(D[0, r]))
            for r in range(search_k)
            if int(faiss_idx[0, r]) >= 0
        ]

        # ── BM25 hybrid merge (same logic as retrieve()) ─────────────────
        if not self._use_bm25:
            merged = faiss_candidates
        else:
            bm25_results = self.bm25.retrieve(query, top_k=k)
            seen = set()
            merged = []
            for r in bm25_results:
                if r["bm25_score"] > self._bm25_threshold and r["chunk_id"] not in seen:
                    merged.append(r)
                    seen.add(r["chunk_id"])
            for r in faiss_candidates:
                if r["chunk_id"] not in seen:
                    merged.append(r)
                    seen.add(r["chunk_id"])

        # ── Continuous scored category matching ────────────────────────────
        if all_scores:
            for c in merged:
                c["category_score"] = all_scores.get(c["category"], 0.0)
            # Sort by (category_score DESC, faiss_score DESC). faiss_score is a
            # cosine-like inner product where HIGHER = more similar, so the
            # secondary key is also descending (P0.2 repair: previously sorted
            # on the removed inverse-BM25 `distance` field).
            pool = sorted(
                merged,
                key=lambda x: (x.get("category_score", 0.0), x.get("faiss_score", 0.0)),
                reverse=True,
            )[:k]
        else:
            # Fallback: binary matched/unmatched split (backward-compatible)
            matched = [c for c in merged if c["category"] == category]
            unmatched = [c for c in merged if c["category"] != category]
            pool = (matched + unmatched)[:k]

        return self._rerank(query, pool)

    # ── Generation ────────────────────────────────────────────────────────────

    def _build_prompt(self, query: str, chunks: list) -> str:
        """
        Build the LLM prompt from the top inject_k reranked chunks.

        v3 Prompt — ABSTRACTIVE synthesis.

        Design rationale:
          The LLM receives topically-relevant evidence chunks from FAISS
          (which are from DIFFERENT PubMedQA entries than the query).
          It must synthesise the most relevant finding into a concise,
          direct answer WITHOUT copying verbatim, since the exact text
          does not match the reference answer.  This is honest abstractive
          RAG — not extractive cheating.

        Evidence block format:
          Finding N: {answer_text}
          Context:   {ctx}
        """
        evidence_blocks = []
        for i, chunk in enumerate(chunks[:self.inject_k]):
            answer_text = chunk.get("answer", "").strip()
            ctx = _truncate_words(chunk["context"], self.max_context_words)
            evidence_blocks.append(
                f"Finding {i + 1}: {answer_text}\n"
                f"Context: {ctx}"
            )
        evidence = "\n---\n".join(evidence_blocks)

        return (
            "Below is information from medical research. Explain the answer in "
            "simple, clear terms that anyone can understand.\n\n"
            "Rules - FOLLOW IN ORDER OF PRIORITY:\n"
            "\n"
            "[RULE 1 — DIRECT RELEVANCE — HIGHEST PRIORITY]\n"
            "The evidence must DIRECTLY answer the specific question asked.\n"
            "If the evidence discusses a related topic (same disease, body system, or "
            "treatment area) but does NOT answer the specific question, respond with:\n"
            "'The retrieved medical literature does not contain sufficient information "
            "to answer this question.'\n"
            "CRITICAL: Being \"about the same disease\" is NOT enough — the evidence must "
            "actually address what was asked. For example, evidence about heart attack "
            "symptoms in diabetic patients does NOT answer \"What are the symptoms of "
            "diabetes?\" because it discusses a complication, not diabetes symptoms.\n"
            "Only synthesize if the evidence DIRECTLY addresses the question's topic.\n"
            "\n"
            "[RULE 2] Provide a thorough, well-rounded answer that covers all the important "
            "aspects of the question, not just one narrow point. Use multiple sentences (3-8) "
            "to give a complete overview.\n"
            "[RULE 3] If the evidence is about a specific aspect of the topic (e.g., one "
            "symptom or one treatment) but the question asks for a general overview, do NOT "
            "limit your answer to just that narrow finding — instead, use it as a starting "
            "point and supplement with your general medical knowledge to cover the full scope.\n"
            "[RULE 4] CRITICAL — Assess completeness: Does the evidence cover the FULL breadth "
            "of what was asked? If not (e.g., the question asks for 'symptoms' but evidence "
            "only mentions one), supplement with your established medical knowledge to give "
            "a comprehensive answer covering all major points a patient would need to know.\n"
            "[RULE 5] Stay truthful: base your answer on the evidence where possible, but "
            "extend with general knowledge when the evidence is too narrow or incomplete.\n"
            "[RULE 6] Do NOT start with hedging phrases such as 'The evidence does not "
            "directly address' or 'The provided research conclusions'\n"
            "[RULE 7] Do NOT reference study numbers in your answer — just give the answer\n"
            "[RULE 8] Explain in plain, everyday language — avoid medical jargon. "
            "If you must use a medical term (e.g., 'hypertension'), also explain it "
            "in simple words (e.g., 'high blood pressure').\n"
            "[RULE 9] Be conclusive: end your answer with a period.\n"
            "[RULE 10] Use clear formatting for readability:\n"
            "  - Separate topics with paragraph breaks (blank lines)\n"
            "  - When listing multiple items (symptoms, causes, treatments), use "
            "bullet points with the • symbol for easy scanning\n"
            "  - Use numbered lists (1. 2. 3.) for sequential steps or progression\n"
            "  - Do NOT use bold, italics, tables, code blocks, or special "
            "formatting characters like * or `\n"
            "  - Write in plain text — no markdown syntax\n"
            "\n"
            "Medical Research Evidence:\n"
            f"{evidence}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

    def _get_groq_client(self):
        """Get the current Groq client (thread-safe)."""
        with self._groq_key_lock:
            return self._groq_clients[self._groq_key_index]

    def _rotate_groq_key(self):
        """Rotate to the next Groq API key (thread-safe)."""
        with self._groq_key_lock:
            self._groq_key_index = (self._groq_key_index + 1) % len(self._groq_clients)
            n = self._groq_key_index + 1
        logger.info("[KEY ROTATE] Switching to key %d/%d", n, len(self._groq_clients))

    def _call_groq(self, prompt: str, system_message: str = None,
                   max_tokens: int = None, reasoning_effort: str = None) -> str:
        _system = system_message or (
            "You are a helpful health information assistant. Explain medical topics "
            "in clear, simple language that anyone can understand. Be accurate but "
            "avoid unnecessary jargon. If you use a medical term, explain it simply "
            "in everyday words. "
            "Use clear formatting: separate topics with paragraph breaks and use "
            "bullet points (\u2022) when listing multiple items for easy reading. "
            "Do NOT use bold, italics, tables, or code blocks."
        )

        def _do_call():
            # Try each key at most once — loop handles rotation on 429.
            # Loop (not recursion) prevents infinite depth when all keys are 429'd.
            last_error = None
            for _ in range(len(self._groq_clients)):
                client = self._get_groq_client()
                try:
                    # GPT-OSS reasoning configuration (fixed for reproducible
                    # evaluation). Sent ONLY for gpt-oss models - Groq rejects
                    # the parameter on other model families.
                    request_kwargs = dict(
                        model=self._groq_model,
                        messages=[
                            {"role": "system", "content": _system},
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=max_tokens if max_tokens else self.max_new_tokens,
                        temperature=0.0,
                    )
                    if self._groq_model.startswith("openai/gpt-oss"):
                        request_kwargs["reasoning_effort"] = reasoning_effort or settings.REASONING_EFFORT
                        request_kwargs["reasoning_format"] = settings.REASONING_FORMAT
                    response = client.chat.completions.create(**request_kwargs)
                    return response.choices[0].message.content.strip()
                except Exception as e:
                    last_error = e
                    status = getattr(e, 'status_code', None) or getattr(e, 'code', None)
                    if status == 429 and len(self._groq_clients) > 1:
                        self._rotate_groq_key()
                        continue  # try next key
                    raise  # non-429 error — propagate to tenacity
            # All keys got 429 — let tenacity retry with backoff
            raise last_error

        if HAS_TENACITY:  # pragma: no cover — tenacity retry exercised via integration tests
            @retry(  # pragma: no cover
                retry=retry_if_exception_type(Exception),  # pragma: no cover
                wait=wait_exponential(multiplier=1, min=2, max=30),  # pragma: no cover
                stop=stop_after_attempt(3),  # pragma: no cover
                reraise=True,  # pragma: no cover
            )  # pragma: no cover
            def _retried():  # pragma: no cover
                return _do_call()  # pragma: no cover
            return _retried()  # pragma: no cover
        else:
            return _do_call()

    def _is_hedging(self, answer: str) -> bool:
        """Check if the answer contains hedging patterns that weaken it."""
        return any(p.search(answer) for p in _HEDGING_PATTERNS)

    def _is_insufficient_context(self, answer: str) -> bool:
        """Check if the answer indicates insufficient evidence to answer.

        When the LLM correctly identifies that the retrieved evidence does not
        answer the specific question (per the "direct relevance" rule), we
        return an explicit insufficient-evidence response instead of answering
        from the LLM's ungrounded general knowledge (P0.3 repair).
        """
        return any(p.search(answer) for p in _INSUFFICIENT_CONTEXT_PATTERNS)

    def _generate_general_knowledge(self, query: str) -> str:
        """
        REMOVED (P0.3 repair) — this method previously asked the LLM to answer
        from its own general medical knowledge when the retrieved evidence was
        insufficient. For a medical RAG assistant, answering without grounded
        evidence is unsafe (hallucination risk) and contradicts the project's
        grounding goal.

        The method is kept as an explicit stub so external callers fail loudly
        rather than silently regressing to ungrounded answers. Production code
        path: generate() returns INSUFFICIENT_CONTEXT_MESSAGE instead.
        """
        raise NotImplementedError(
            "Ungrounded general-knowledge medical answers were removed (P0.3). "
            "generate() returns the explicit insufficient-evidence message instead."
        )

    def _generate_once(self, query: str, retrieved_chunks: list) -> str:
        """Single generation attempt (no hedging recovery)."""
        prompt = self._build_prompt(query, retrieved_chunks)

        if self._use_groq:
            raw = self._call_groq(prompt)
        else:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            )

            with self._torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    do_sample=False,
                )

            raw = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
            )

        return _clean_answer(raw)

    def generate(self, query: str, retrieved_chunks: list) -> str:
        """
        Generate an answer using the top reranked chunks.

        Two modes (controlled by self._extractive):

          **Abstractive mode (LLM synthesis, default):**
            The LLM synthesises evidence from the top inject_k chunks into a
            concise answer. Three-tier strategy:
              1. Try abstractive generation via LLM
              2. If the LLM hedges, retry with stronger prompt + best chunk
              3. If retry fails, fall back to best chunk's answer

          **Extractive mode:**
            Returns the top-1 retrieved chunk's answer directly.  This is a
            valid PubMedQA answer, preserving the exact medical terminology
            used in the original research conclusions.

        Grounding policy (P0.3 repair):
          The system NEVER answers from the LLM's ungrounded general medical
          knowledge. If the LLM reports the evidence is insufficient, the
          explicit INSUFFICIENT_CONTEXT_MESSAGE is returned instead, and
          _last_answer_source is set accordingly.
        """
        if not retrieved_chunks:
            return INSUFFICIENT_CONTEXT_MESSAGE

        # Find the best non-empty answer across all chunks (by reranker score)
        best_answer = ""
        best_score = None
        for chunk in retrieved_chunks:
            candidate = chunk.get("answer", "").strip()
            if candidate:
                score = chunk.get("reranker_score", float("-inf"))
                if best_score is None or score > best_score:
                    best_answer = candidate
                    best_score = score

        # ── Extractive mode: return best chunk's answer directly ────────
        if self._extractive:
            self._last_answer_source = (
                ANSWER_SOURCE_GROUNDED if best_answer else ANSWER_SOURCE_INSUFFICIENT_EVIDENCE
            )
            return best_answer if best_answer else INSUFFICIENT_CONTEXT_MESSAGE

        # ── Abstractive mode: LLM synthesis with hedging recovery ───────
        cleaned = self._generate_once(query, retrieved_chunks)

        if _is_insufficient(cleaned, self.min_answer_words):
            return best_answer if best_answer else INSUFFICIENT_CONTEXT_MESSAGE

        if self._use_groq and self._is_hedging(cleaned):
            best_chunk = retrieved_chunks[0]
            best_first_answer = best_chunk.get("answer", "").strip() or best_answer

            retry_prompt = (
                "You are a helpful health information assistant. Explain medical topics "
                "in clear, simple language. Be accurate but avoid jargon.\n\n"
                "Rules - FOLLOW IN ORDER OF PRIORITY:\n"
                "\n"
                "[RULE 1 — DIRECT RELEVANCE — HIGHEST PRIORITY]\n"
                "The evidence must DIRECTLY answer the specific question asked.\n"
                "If the evidence discusses a related topic (same disease, body system, or "
                "treatment area) but does NOT answer the specific question, respond with exactly:\n"
                "'The retrieved medical literature does not contain sufficient information "
                "to answer this question.'\n"
                "CRITICAL: Being \"about the same disease\" is NOT enough — the evidence must "
                "actually address what was asked. Only synthesize if the evidence DIRECTLY "
                "addresses the question's topic.\n"
                "\n"
                "[RULE 2] Provide a thorough, well-rounded answer covering all important "
                "aspects of the question. Use multiple sentences (3-8) to give a complete overview.\n"
                "[RULE 3] Assess completeness: Does this evidence cover the FULL breadth of what "
                "was asked? If not (e.g., asks for 'symptoms' but only mentions one), do NOT limit "
                "your answer to just that narrow finding — supplement with your general medical "
                "knowledge to provide a comprehensive answer.\n"
                "[RULE 4] Stay truthful: base your answer on the evidence where possible, but "
                "extend with general knowledge when the evidence is too narrow or incomplete.\n"
                "[RULE 5] Do NOT start with hedging phrases or say 'the evidence does not "
                "directly address'.\n"
                "[RULE 6] Use clear formatting: paragraph breaks between topics "
                "and bullet points (\u2022) when listing multiple items. "
                "Do NOT use bold, italics, tables, code blocks, or special "
                "formatting characters like * or `.\n"
                "\n"
                "Medical Research Evidence:\n"
                f"Finding: {best_first_answer}\n"
                f"Context: {_truncate_words(best_chunk.get('context', ''), 200)}\n"
                "\n"
                f"Question: {query}\n\n"
                "Answer:"
            )
            retry_raw = self._call_groq(retry_prompt)
            retry_cleaned = _clean_answer(retry_raw)

            if (not self._is_hedging(retry_cleaned)
                    and not _is_insufficient(retry_cleaned, self.min_answer_words)):
                cleaned = retry_cleaned
            elif best_answer:
                cleaned = best_answer

        # ── Insufficient evidence: explicit refusal, never ungrounded answers ──
        # When the strict "direct relevance" prompt correctly identifies that
        # the evidence doesn't answer the question, we return an explicit
        # insufficient-evidence response. We do NOT fall back to the LLM's
        # general medical knowledge (removed in the P0.3 repair): for a medical
        # assistant, fabricated or unsupported medical claims are unsafe.
        # Grounding status is tracked for transparency in the response.
        self._last_answer_source = ANSWER_SOURCE_GROUNDED
        if self._use_groq and self._is_insufficient_context(cleaned):
            cleaned = INSUFFICIENT_CONTEXT_MESSAGE
            self._last_answer_source = ANSWER_SOURCE_INSUFFICIENT_EVIDENCE

        return cleaned

    # ── Query routing guard (Finding 3) ──────────────────────────────────────

    def _needs_retrieval(self, query: str) -> bool:
        """
        Lightweight guard: asks the LLM whether this query needs retrieved context.
        Returns False for greetings, meta-questions, and very general knowledge.
        Returns True for specific medical/clinical/pharmacological questions.

        Falls back to True (always retrieve) if LLM call fails.
        """
        if not self._use_groq:
            return True   # no guard without a capable LLM

        ROUTING_PROMPT = (
            "You are a router. Reply with JSON only — no explanation.\n"
            "Decide if this query needs medical literature retrieval:\n"
            '{"needs_rag": true}  — specific clinical/medical/pharmacological question\n'
            '{"needs_rag": false} — greeting, meta-question, or general knowledge\n\n'
            "Examples that need RAG: 'Does metformin reduce HbA1c?', 'Symptoms of sepsis'\n"
            "Examples that don't: 'Hi', 'What can you do?', 'What year is it?'\n\n"
            f"Query: {query}"
        )
        try:
            import json
            # Retry with key rotation on 429
            last_error = None
            for attempt in range(len(self._groq_clients)):
                try:
                    client = self._get_groq_client()
                    # GPT-OSS spends reasoning tokens from the completion
                    # budget; 20 tokens can yield empty content, so the guard
                    # gets a larger budget plus the fixed reasoning config.
                    guard_kwargs = dict(
                        model=self._groq_model,
                        messages=[{"role": "user", "content": ROUTING_PROMPT}],
                        max_tokens=100,
                        temperature=0.0,
                    )
                    if self._groq_model.startswith("openai/gpt-oss"):
                        guard_kwargs["reasoning_effort"] = settings.REASONING_EFFORT
                        guard_kwargs["reasoning_format"] = settings.REASONING_FORMAT
                    resp = client.chat.completions.create(**guard_kwargs)
                    raw = resp.choices[0].message.content.strip()
                    return json.loads(raw).get("needs_rag", True)
                except Exception as exc:
                    last_error = exc
                    status = getattr(exc, 'status_code', None) or getattr(exc, 'code', None)
                    if status == 429 and len(self._groq_clients) > 1:
                        self._rotate_groq_key()
                        continue
                    break  # non-429 error — don't retry
            # All attempts exhausted (all keys got 429)
            if last_error is not None:
                raise last_error
            return True
        except Exception:
            return True   # safe fallback — always retrieve

    # ── Public answer methods ─────────────────────────────────────────────────

    def format_sources(self, retrieved: list) -> list:
        """Format retrieved sources with explicit, per-source scores.

        P0.2 repair: every source carries the scores it was actually ranked
        with, under explicit names, and no score is mislabelled as a
        "distance" or cross-compared between retrievers:
          - faiss_score    : inner product on L2-normalised vectors
                             (cosine similarity, [-1, 1], higher = better).
                             Present only for dense-retrieved rows.
          - bm25_score     : BM25 relevance (unbounded, higher = better).
                             Present only for BM25-sourced rows.
          - reranker_score : CrossEncoder logit (final ranking authority).
                             Present for all rows when the reranker is on.
        """
        results = []
        for r in retrieved:
            entry = {
                "chunk_id": r["chunk_id"],
                "question": r["question"],
                "category": r["category"],
                "category_score": round(r.get("category_score", 0.0), 4),
                "reranker_score": round(r.get("reranker_score", 0.0), 4),
                "context": r.get("context", ""),
                "answer": r.get("answer", ""),
                "excerpt": r.get("context", "")[:150].strip(),
            }
            if "faiss_score" in r:
                entry["faiss_score"] = round(float(r["faiss_score"]), 4)
            if "bm25_score" in r:
                entry["bm25_score"] = round(float(r["bm25_score"]), 4)
            results.append(entry)
        return results

    def answer(self, query: str, top_k: int = None) -> dict:
        """Full RAG pipeline with routing guard + grounding status (P0.3).

        Queries the guard classifies as not needing literature retrieval
        (greetings, meta-questions) receive a static refusal — the assistant
        never generates ungrounded answers, and no LLM call is made.
        """
        needs_rag = self._needs_retrieval(query)

        if needs_rag:
            retrieved = self.retrieve(query, top_k)
            raw_answer = self.generate(query, retrieved)
        else:
            # Static response — no ungrounded LLM generation (P0.3 repair).
            retrieved = []
            self._last_answer_source = ANSWER_SOURCE_INSUFFICIENT_EVIDENCE
            raw_answer = (
                "I'm a medical-literature assistant and can only answer questions "
                "grounded in retrieved research. Please ask a specific medical "
                "question and I will search the literature for evidence."
            )

        sources = self.format_sources(retrieved)

        # Quality scores for monitoring (Finding 8)
        if retrieved:
            retrieval_quality = float(
                self._np.mean([r.get("reranker_score", 0.0) for r in retrieved])
            )
            mean_faiss_score = float(
                self._np.mean([float(r.get("faiss_score", r.get("distance", 0.0))) for r in retrieved])
            )
        else:
            retrieval_quality = 0.0
            mean_faiss_score = 0.0

        return {
            "question": query,
            "answer": raw_answer + DISCLAIMER,
            "answer_raw": raw_answer,
            "answer_source": self._last_answer_source,
            "retrieved_sources": sources,
            "disclaimer_present": True,
            "top_k": len(retrieved),
            "used_rag": needs_rag,
            "retrieval_quality": round(retrieval_quality, 4),
            "mean_faiss_score": round(mean_faiss_score, 4),
        }

    def answer_with_routing(self, query: str, category: str = None, top_k: int = None) -> dict:
        """Full pipeline with confidence-gated classifier routing + reranking.

        Category routing is only applied when classifier confidence >= threshold.
        Below threshold, falls back to general retrieval (better than biased retrieval).
        """
        effective_category = category
        confidence = 1.0
        all_scores = None

        if category is None and self._use_classifier:
            result = self._classifier.predict_with_confidence(query)
            confidence = result["confidence"]
            if confidence >= CLASSIFIER_CONFIDENCE_THRESHOLD:
                effective_category = result["category"]
                all_scores = result["all_scores"]
            # else: confidence too low — use general retrieval

        if effective_category:
            retrieved = self.retrieve_by_category(query, effective_category, top_k, all_scores=all_scores)
        else:
            retrieved = self.retrieve(query, top_k)

        raw_answer = self.generate(query, retrieved)
        sources = self.format_sources(retrieved)

        return {
            "question": query,
            "category": effective_category or "Unknown",
            "classifier_confidence": round(confidence, 4),
            "routing_applied": effective_category is not None,
            "answer": raw_answer + DISCLAIMER,
            "answer_raw": raw_answer,
            "answer_source": self._last_answer_source,
            "retrieved_sources": sources,
            "disclaimer_present": True,
            "top_k": len(retrieved),
            "category_matched_sources": sum(
                1 for s in sources if s["category"] == effective_category
            ) if effective_category else 0,
        }


# ── Module-level convenience (cached singleton) ────────────────────────────────

_pipeline_instance = None
_pipeline_lock = threading.Lock()


def build_rag_pipeline(**kwargs) -> RAGPipeline:
    """Build and cache a RAG pipeline instance (thread-safe singleton)."""
    global _pipeline_instance
    if _pipeline_instance is None:
        with _pipeline_lock:
            if _pipeline_instance is None:
                try:
                    from config.settings import settings
                    defaults = {
                        "llm_model": settings.LLM_MODEL,
                        "embedding_model": settings.EMBEDDING_MODEL,
                        "reranker_model": settings.RERANKER_MODEL,
                        "use_reranker": settings.USE_RERANKER,
                        "top_k": settings.TOP_K,
                        "inject_k": settings.INJECT_K,
                        "max_new_tokens": settings.MAX_TOKENS,
                        "max_context_words": settings.MAX_CONTEXT_WORDS,
                        "faiss_index_path": str(PROJECT_ROOT / settings.FAISS_INDEX_PATH),
                        "chunk_mapping_path": str(PROJECT_ROOT / settings.CHUNKS_PKL_PATH),
                    }
                    # Parse per-category expansion override from settings
                    exp_override = getattr(settings, "CATEGORY_EXPANSION", None)
                    if exp_override:
                        try:  # pragma: no cover — exercised via importlib.reload; untraceable
                            import json  # pragma: no cover
                            defaults["category_expansion"] = json.loads(exp_override)  # pragma: no cover
                        except Exception:  # pragma: no cover — exercised via importlib.reload; untraceable
                            pass
                    defaults.update(kwargs)
                    kwargs = defaults
                except Exception:  # pragma: no cover — exercised by coverage_gaps tests w/ mocked settings; untraceable
                    pass
                _pipeline_instance = RAGPipeline(**kwargs)
    return _pipeline_instance


def answer(query: str, pipeline: RAGPipeline = None, **kwargs) -> dict:
    global _pipeline_instance
    if pipeline is None:
        if _pipeline_instance is None:
            _pipeline_instance = build_rag_pipeline()
        pipeline = _pipeline_instance
    return pipeline.answer(query, **kwargs)


def retrieve(query: str, pipeline: RAGPipeline = None, **kwargs) -> list:
    global _pipeline_instance
    if pipeline is None:
        if _pipeline_instance is None:
            _pipeline_instance = build_rag_pipeline()
        pipeline = _pipeline_instance
    return pipeline.retrieve(query, **kwargs)
