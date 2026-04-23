# src/rag/pipeline.py
# Purpose: the production RAG chain.
# Takes a query, retrieves top-5 relevant chunks from FAISS,
# injects them as context into an LLM prompt, and returns
# the grounded answer with a mandatory medical disclaimer.

import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# File paths — relative to repo root. Keep these fixed.
FAISS_INDEX_PATH = "data/embeddings/faiss_index/pubmedqa_index_flatl2.faiss"
CHUNKS_PATH = "data/embeddings/faiss_index/chunk_mapping.pkl"

DISCLAIMER = (
    "\n\n⚠️ MEDICAL DISCLAIMER: This response is for informational purposes only. "
    "Always consult a qualified healthcare professional for medical advice, "
    "diagnosis, or treatment."
)

# Load embedding model once
_embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index and chunks once
_index = None
_chunks = None

def _load_retrieval():
    global _index, _chunks
    if _index is None:
        _index = faiss.read_index(FAISS_INDEX_PATH)
        with open(CHUNKS_PATH, "rb") as f:
            _chunks = pickle.load(f)

# LLM client — created lazily on first use, not at import time
_llm = None


def _get_llm():
    """Return the LLM client, creating it only on first call."""
    global _llm
    if _llm is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY environment variable is not set. "
                "Add it to your .env file and load it before calling generate()."
            )
        _llm = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )
    return _llm


def retrieve(query: str, k: int = 5) -> list[str]:
    _load_retrieval()
    query_vector = _embed_model.encode([query])
    distances, indices = _index.search(query_vector, k)
    return [_chunks['text_chunk'].iloc[i] for i in indices[0]]


def generate(query: str, context_chunks: list[str]) -> str:
    client = _get_llm()   # ← use this instead of _llm directly
    context = "\n\n".join(context_chunks)
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a medical information assistant. "
                    "Answer the question using ONLY the provided context. "
                    "If the context does not contain enough information, say so honestly."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}"
            }
        ]
    )
    return response.choices[0].message.content


def answer(query: str) -> dict:
    """
    Full RAG answer: retrieve → generate → add disclaimer.
    Returns a dict (ready for the FastAPI endpoint in M3).
    
    Args:
        query: user's medical question
    
    Returns:
        dict with keys: answer, retrieved_sources, disclaimer_present
    """
    chunks = retrieve(query)
    raw_answer = generate(query, chunks)
    final_answer = raw_answer + DISCLAIMER
    
    return {
        "answer": final_answer,
        "retrieved_sources": chunks,
        "disclaimer_present": True
    }