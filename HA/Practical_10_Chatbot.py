import requests
import json
import codecs

EMBEDDING_MODEL = "llama3"
LANGUAGE_MODEL = "llama3"

# Function to call Ollama's embedding endpoint
def get_embedding(text):
    response = requests.post("http://localhost:11434/api/embeddings", json={
        "model": EMBEDDING_MODEL,
        "prompt": text
    })
    return response.json()["embedding"]

# Function to chat with Ollama
def chat_with_ollama(system_prompt, user_prompt):
    response = requests.post("http://localhost:11434/api/chat", json={
        "model": LANGUAGE_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }, stream=True)

    for line in response.iter_lines():
        if line:
            msg = json.loads(line.decode("utf-8"))
            print(msg["message"]["content"], end='', flush=True)

# Load data
with codecs.open("health.txt", "r", encoding="utf-8", errors="ignore") as f:
    dataset = f.readlines()
    print(f"✅ Loaded {len(dataset)} health facts")

# Build vector database
VECTOR_DB = []
for i, chunk in enumerate(dataset):
    emb = get_embedding(chunk)
    VECTOR_DB.append((chunk.strip(), emb))
    print(f"🔹 Indexed {i+1}/{len(dataset)}")

# Cosine similarity
def cosine_similarity(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    norm_a = sum(x**2 for x in a)**0.5
    norm_b = sum(x**2 for x in b)**0.5
    return dot / (norm_a * norm_b)

# Retrieval
def retrieve(query, top_n=3):
    q_embed = get_embedding(query)
    scored = [(chunk, cosine_similarity(q_embed, emb)) for chunk, emb in VECTOR_DB]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]

# Chat loop
while True:
    user_query = input("\n💬 Ask a health-related question (or type 'exit' to quit): ")
    if user_query.lower() == "exit":
        break

    results = retrieve(user_query)
    print("\n📚 Retrieved facts:")
    for chunk, score in results:
        print(f" - ({score:.2f}) {chunk}")

    context = " ".join(f"- {chunk}" for chunk, _ in results)
    prompt = f"You are a helpful health assistant. Use only the following facts to answer the user's question:\n{context}"

    print("\n🤖 Chatbot says:")
    chat_with_ollama(prompt, user_query)
