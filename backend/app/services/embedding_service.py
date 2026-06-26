from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)


def generate_embeddings(chunks: list[dict]) -> list[dict]:
    """
    Generate embeddings for text chunks using a sentence-transformers model.
    """

    if not chunks:
        return []

    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts)

    embedded_chunks = []

    for chunk, embedding in zip(chunks, embeddings):
        embedded_chunks.append(
            {
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "start_char": chunk["start_char"],
                "end_char": chunk["end_char"],
                "embedding": embedding.tolist()
            }
        )

    return embedded_chunks


def generate_query_embedding(question: str) -> list[float]:
    """
    Generate an embedding for a user question.
    """

    embedding = model.encode(question)

    return embedding.tolist()