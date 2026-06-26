def generate_grounded_answer(question: str, retrieved_chunks: list[dict]) -> dict:
    """
    Generate a simple grounded answer using retrieved document chunks.

    This version avoids copying very long raw chunks directly.
    It returns a cleaner answer with source references.
    """

    if not retrieved_chunks:
        return {
            "answer": "I could not find relevant information in the uploaded document.",
            "sources": []
        }

    source_summaries = []
    answer_points = []

    for index, chunk in enumerate(retrieved_chunks, start=1):
        text = chunk.get("text", "").strip()
        metadata = chunk.get("metadata", {})

        if not text:
            continue

        cleaned_text = " ".join(text.split())

        short_text = cleaned_text[:350]

        answer_points.append(
            f"[Source {index}] {short_text}..."
        )

        source_summaries.append(
            {
                "source_number": index,
                "citation": f"[Source {index}: {metadata.get('filename')}, chunk {metadata.get('chunk_id')}]",
                "filename": metadata.get("filename"),
                "chunk_id": metadata.get("chunk_id"),
                "start_char": metadata.get("start_char"),
                "end_char": metadata.get("end_char"),
                "distance": chunk.get("distance")
            }
        )

    final_answer = (
        "Based on the retrieved sections, the document appears to discuss the following points:\n\n"
        + "\n\n".join(answer_points)
    )

    return {
        "answer": final_answer,
        "sources": source_summaries
    }