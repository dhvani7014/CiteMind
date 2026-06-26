import chromadb


CHROMA_DB_PATH = "backend/chroma_db"
COLLECTION_NAME = "citemind_papers"

client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)


def store_chunks_in_vector_db(embedded_chunks: list[dict], filename: str) -> dict:
    """
    Store embedded text chunks in ChromaDB.
    """

    if not embedded_chunks:
        return {
            "status": "empty",
            "stored_chunks": 0,
            "collection_name": COLLECTION_NAME
        }

    ids = []
    documents = []
    embeddings = []
    metadatas = []

    for chunk in embedded_chunks:
        chunk_id = f"{filename}_chunk_{chunk['chunk_id']}"

        ids.append(chunk_id)
        documents.append(chunk["text"])
        embeddings.append(chunk["embedding"])
        metadatas.append(
            {
                "filename": filename,
                "chunk_id": chunk["chunk_id"],
                "start_char": chunk["start_char"],
                "end_char": chunk["end_char"]
            }
        )

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return {
        "status": "stored",
        "stored_chunks": len(ids),
        "collection_name": COLLECTION_NAME
    }


def search_similar_chunks(query_embedding: list[float], top_k: int = 3) -> list[dict]:
    """
    Search ChromaDB for chunks most similar to the user question.
    """

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    retrieved_chunks = []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for document, metadata, distance in zip(documents, metadatas, distances):
        retrieved_chunks.append(
            {
                "text": document,
                "metadata": metadata,
                "distance": distance
            }
        )

    return retrieved_chunks


def get_collection_count() -> int:
    """
    Return the number of chunks currently stored in the vector database.
    """

    return collection.count()