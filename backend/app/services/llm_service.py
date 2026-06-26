import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()


GROQ_MODEL = "llama-3.1-8b-instant"


def build_context_from_chunks(retrieved_chunks: list[dict]) -> str:
    """
    Build a compact context string from retrieved chunks.
    """

    context_parts = []

    for index, chunk in enumerate(retrieved_chunks, start=1):
        text = chunk.get("text", "").strip()
        metadata = chunk.get("metadata", {})

        filename = metadata.get("filename", "unknown_file")
        chunk_id = metadata.get("chunk_id", "unknown_chunk")

        citation = f"[Source {index}: {filename}, chunk {chunk_id}]"

        context_parts.append(
            f"{citation}\n{text}"
        )

    return "\n\n".join(context_parts)


def generate_llm_answer(question: str, retrieved_chunks: list[dict]) -> dict | None:
    """
    Generate an LLM-based answer using Groq if GROQ_API_KEY exists.

    Returns None if no API key is available, so the app can fallback safely.
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return None

    if not retrieved_chunks:
        return {
            "answer": "I could not find relevant information in the uploaded document.",
            "llm_provider": "groq",
            "llm_model": GROQ_MODEL
        }

    client = Groq(api_key=api_key)

    context = build_context_from_chunks(retrieved_chunks)

    system_prompt = """
You are CiteMind, a citation-grounded AI research paper assistant.

Answer the user's question using only the provided context.
Do not use outside knowledge.
If the context does not contain enough information, say that clearly.
Include citations in the answer using the exact source labels provided, such as [Source 1: paper.pdf, chunk 4].
Keep the answer clear, concise, and useful.
"""

    user_prompt = f"""
Question:
{question}

Retrieved context:
{context}
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.2,
        max_tokens=700
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "llm_provider": "groq",
        "llm_model": GROQ_MODEL
    }