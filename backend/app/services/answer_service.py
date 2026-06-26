def format_citation(source_number: int, metadata: dict) -> str:
    """
    Create a clean citation label for a retrieved chunk.
    """

    filename = metadata.get("filename", "unknown_file")
    chunk_id = metadata.get("chunk_id", "unknown_chunk")

    return f"[Source {source_number}: {filename}, chunk {chunk_id}]"


def clean_text(text: str, max_length: int = 400) -> str:
    """
    Clean and shorten retrieved text for readable output.
    """

    cleaned = " ".join(text.split())

    if len(cleaned) <= max_length:
        return cleaned

    return cleaned[:max_length].rstrip() + "..."


def generate_grounded_answer(question: str, retrieved_chunks: list[dict]) -> dict:
    """
    Generate a simple citation-grounded answer using retrieved document chunks.

    This version formats sources clearly and keeps the answer readable.
    """

    if not retrieved_chunks:
        return {
            "answer": "I could not find relevant information in the uploaded document.",
            "citations": [],
            "sources": []
        }

    answer_points = []
    citations = []
    source_summaries = []

    for index, chunk in enumerate(retrieved_chunks, start=1):
        text = chunk.get("text", "").strip()
        metadata = chunk.get("metadata", {})

        if not text:
            continue

        citation = format_citation(index, metadata)
        short_text = clean_text(text)

        answer_points.append(
            f"{citation} {short_text}"
        )

        citations.append(citation)

        source_summaries.append(
            {
                "source_number": index,
                "citation": citation,
                "filename": metadata.get("filename"),
                "chunk_id": metadata.get("chunk_id"),
                "start_char": metadata.get("start_char"),
                "end_char": metadata.get("end_char"),
                "distance": chunk.get("distance")
            }
        )

    final_answer = (
        "Based on the retrieved sections, here is the most relevant information:\n\n"
        + "\n\n".join(answer_points)
    )

    return {
        "answer": final_answer,
        "citations": citations,
        "sources": source_summaries
    }