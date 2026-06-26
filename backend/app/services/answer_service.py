def format_citation(source_number: int, metadata: dict) -> str:
    """
    Create a clean citation label for a retrieved chunk.
    """

    filename = metadata.get("filename", "unknown_file")
    chunk_id = metadata.get("chunk_id", "unknown_chunk")

    return f"[Source {source_number}: {filename}, chunk {chunk_id}]"


def clean_text(text: str, max_length: int = 550) -> str:
    """
    Clean and shorten retrieved text for readable output.
    """

    cleaned = " ".join(text.split())

    if len(cleaned) <= max_length:
        return cleaned

    return cleaned[:max_length].rstrip() + "..."


def create_source_summary(
    source_number: int,
    citation: str,
    metadata: dict,
    chunk: dict,
) -> dict:
    """
    Create clean source metadata for frontend display.
    """

    return {
        "source_number": source_number,
        "citation": citation,
        "filename": metadata.get("filename"),
        "chunk_id": metadata.get("chunk_id"),
        "start_char": metadata.get("start_char"),
        "end_char": metadata.get("end_char"),
        "distance": chunk.get("distance"),
    }


def generate_grounded_answer(question: str, retrieved_chunks: list[dict]) -> dict:
    """
    Generate a readable fallback answer using retrieved document chunks.

    This is used when the LLM provider is unavailable or fails.
    It does not generate new reasoning like an LLM, but it still gives
    useful paper-grounded information with citations.
    """

    if not retrieved_chunks:
        return {
            "answer": (
                "I could not find enough relevant information in the uploaded document "
                "to answer that confidently."
            ),
            "citations": [],
            "sources": [],
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
            f"- {short_text} {citation}"
        )

        citations.append(citation)

        source_summaries.append(
            create_source_summary(
                source_number=index,
                citation=citation,
                metadata=metadata,
                chunk=chunk,
            )
        )

    if not answer_points:
        return {
            "answer": (
                "I found matching document sections, but they did not contain readable text "
                "that could be used to answer your question."
            ),
            "citations": [],
            "sources": [],
        }

    final_answer = (
        "I could not generate a full AI answer right now, but I found these relevant "
        "sections from the uploaded paper:\n\n"
        + "\n\n".join(answer_points)
        + "\n\nUse these sources to support your answer, or try again when the LLM provider is available."
    )

    return {
        "answer": final_answer,
        "citations": citations,
        "sources": source_summaries,
    }