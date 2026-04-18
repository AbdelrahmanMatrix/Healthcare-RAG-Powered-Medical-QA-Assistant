
import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index
index = faiss.read_index("pubmedqa_index_flatl2.faiss")

# Load text chunks
with open("chunk_mapping.pkl", "rb") as f:
    text_chunks = pickle.load(f)

# Initialize LLM client
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

def retrieve(query, k=3):
    query_vector = model.encode([query])
    D, I = index.search(query_vector, k)
    return [text_chunks[i] for i in I[0]]

def get_answer(query):
    results = retrieve(query)
    context = "\n".join(results)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "Answer only based on the provided context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
    )

    answer = response.choices[0].message.content

    disclaimer = "\n\n⚠️ This system is for educational purposes only."

    return answer + disclaimer


if __name__ == "__main__":
    print(get_answer("What are the symptoms of diabetes?"))
