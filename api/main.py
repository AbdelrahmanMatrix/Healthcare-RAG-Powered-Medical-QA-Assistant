from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import os
from src.rag.pipeline import MedicalRAGPipeline
from src.rag.embeddings import MedicalEmbedder
from src.rag.vectorstore import MedicalVectorStore
from src.classification.classifier import MedicalQueryClassifier

app = FastAPI(title="Healthcare RAG Medical Assistant")

# Global state for components
# In production, these should be initialized properly or via dependency injection
class AppState:
    def __init__(self):
        self.embedder = MedicalEmbedder()
        self.dim = self.embedder.get_embedding_dimension()

        storage_path = "data/embeddings"
        if os.path.exists(os.path.join(storage_path, "index.faiss")):
            print(f"Loading vector store from {storage_path}")
            self.vector_store = MedicalVectorStore.load(storage_path)
        else:
            print("Warning: Vector store not found. Initializing empty one.")
            self.vector_store = MedicalVectorStore(self.dim)

        self.pipeline = MedicalRAGPipeline(self.vector_store, self.embedder)
        self.classifier = MedicalQueryClassifier()

state = None

@app.on_event("startup")
async def startup_event():
    global state
    state = AppState()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    category: str
    response: str
    retrieved_contexts: List[Dict[str, Any]]

@app.get("/")
async def root():
    return {"message": "Welcome to the Healthcare RAG Medical Assistant API"}

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    if not state:
        raise HTTPException(status_code=503, detail="System initializing")

    # 1. Classify
    classification = state.classifier.classify(request.query)

    # 2. RAG
    rag_result = state.pipeline.answer(request.query)

    return {
        "query": request.query,
        "category": classification["category"],
        "response": rag_result["response"],
        "retrieved_contexts": rag_result["retrieved_contexts"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
