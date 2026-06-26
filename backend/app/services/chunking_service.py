def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[dict]:
    """
    Split extracted PDF text into overlapping chunks for RAG.

    chunk_size controls the maximum number of characters per chunk.
    overlap controls how many characters are shared between chunks.
    """

    if not text:
        return []

    chunks = []
    start = 0
    chunk_id = 1

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": chunk,
                    "start_char": start,
                    "end_char": min(end, len(text))
                }
            )
            chunk_id += 1

        start += chunk_size - overlap

    return chunks