import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load Embedding Model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Step 1: Text Chunking
def chunk_text(text, max_chars=500):

    lines = [
        line.strip()
        for line in text.split("\n")
        if line.strip()
    ]

    chunks = []
    current_chunk = ""

    for line in lines:

        if len(current_chunk) + len(line) + 1 <= max_chars:
            current_chunk += " " + line

        else:
            if current_chunk:
                chunks.append(current_chunk.strip())

            current_chunk = line

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# Step 2: Generate Normalized Embeddings
def create_embeddings(chunks):

    embeddings = model.encode(
        chunks,
        normalize_embeddings=True
    )

    return np.array(embeddings).astype("float32")


# Step 3: Create FAISS Vector Database
def create_vector_store(chunks):

    if not chunks:
        raise ValueError("No text chunks available!")

    embeddings = create_embeddings(chunks)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


# Step 4: Semantic Search with Relevance Checking
def search_chunks(query, index, chunks, top_k=3):

    if not query.strip():
        return []

    if index is None or not chunks:
        return []

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.array(
        query_embedding
    ).astype("float32")

    k = min(top_k, len(chunks))

    scores, indices = index.search(
        query_embedding,
        k
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        similarity = float(score)

        if similarity >= 0.35:

            results.append({
                "chunk": chunks[idx],
                "similarity": similarity
            })

    return results