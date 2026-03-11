import os
from typing import List, Dict, Any
from src.rag.embeddings import MedicalEmbedder
from src.rag.vectorstore import MedicalVectorStore
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline as hf_pipeline

class MedicalRAGPipeline:
    def __init__(self, vector_store: MedicalVectorStore, embedder: MedicalEmbedder):
        self.vector_store = vector_store
        self.embedder = embedder

        # Using a small, efficient model for generation
        model_id = "google/flan-t5-small"
        hf_pipe = hf_pipeline("text2text-generation", model=model_id, max_length=512)
        self.llm = HuggingFacePipeline(pipeline=hf_pipe)

        self.template = """
        You are a medical assistant. Use the following pieces of retrieved context to answer the question.
        If the context does not contain enough information, state that you don't know based on the provided data.

        Context: {context}

        Question: {question}

        Answer:
        """
        self.prompt = PromptTemplate(template=self.template, input_variables=["context", "question"])
        self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        query_embedding = self.embedder.generate_embeddings([query])
        return self.vector_store.search(query_embedding, k=k)

    def generate_response(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        if not contexts:
            return "I'm sorry, I couldn't find any relevant medical information to answer your question."

        context_text = "\n\n".join([
            f"Source: {res.get('source', 'unknown')}\nContext: {res.get('context', '')}\nAnswer: {res.get('answer', '')}"
            for res in contexts
        ])

        response = self.llm_chain.run(context=context_text, question=query)

        disclaimer = "\n\nDISCLAIMER: This is an informational assistant only. Always consult a qualified healthcare professional for medical decisions."
        return response + disclaimer

    def answer(self, query: str) -> Dict[str, Any]:
        retrieved_docs = self.retrieve(query)
        response = self.generate_response(query, retrieved_docs)

        return {
            "query": query,
            "response": response,
            "retrieved_contexts": retrieved_docs
        }

if __name__ == "__main__":
    # Mocking for testing
    from src.rag.embeddings import MedicalEmbedder
    from src.rag.vectorstore import MedicalVectorStore
    import numpy as np
    import pandas as pd

    embedder = MedicalEmbedder()
    dim = embedder.get_embedding_dimension()
    vs = MedicalVectorStore(dim)

    # Adding a sample
    text = "Aspirin is used to reduce pain, fever, or inflammation."
    emb = embedder.generate_embeddings([text])
    meta = pd.DataFrame([{"question": "What is aspirin used for?", "answer": text, "source": "test", "context": ""}])
    vs.add_embeddings(emb, meta)

    pipeline = MedicalRAGPipeline(vs, embedder)
    result = pipeline.answer("What are the uses of aspirin?")
    print(f"Query: {result['query']}")
    print(f"Response: {result['response']}")
