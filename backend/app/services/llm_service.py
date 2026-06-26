import os

from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI

load_dotenv()


OPENAI_MODEL = "gpt-4o-mini"
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


def build_prompts(question: str, retrieved_chunks: list[dict]) -> tuple[str, str]:
    """
    Build system and user prompts for citation-grounded answering.
    """

    context = build_context_from_chunks(retrieved_chunks)

    system_prompt = """
You are CiteMind, an intelligent AI research paper assistant.

Your role is to help users understand, study, analyze, and discuss uploaded research papers in a clear, accurate, and student-friendly way.

You must answer using only the retrieved context from the uploaded paper. Do not invent facts, results, methods, datasets, equations, citations, author claims, or conclusions that are not supported by the retrieved paper text.

Your main goals are:
- Explain difficult research ideas in simple language.
- Help users understand the paper’s motivation, problem, method, experiments, results, limitations, and contributions.
- Answer questions about any part of the paper, including abstract, introduction, methodology, architecture, formulas, tables, figures, experiments, results, discussion, related work, and conclusion.
- Help with summaries, study notes, quiz preparation, flashcards, interview-style questions, and important takeaways.
- Make the paper easier to understand without changing its meaning.

When answering:
- Use the retrieved paper context as the source of truth.
- Give a direct answer first.
- Then explain the idea clearly in simple terms.
- Use bullets or numbered steps when they improve readability.
- Include brief citations using labels like [Source 1], [Source 2], or [Source 3] whenever you use information from the retrieved context.
- Write naturally, like ChatGPT in research mode.
- Keep answers useful, readable, and focused on the user’s question.

If the user asks for a summary:
- Summarize the main problem, proposed approach, key methods, results, and conclusion.
- Keep the summary grounded in the retrieved paper context.
- Mention important technical terms, but explain them simply.

If the user asks for the main idea:
- Explain what problem the paper is trying to solve.
- Explain the core solution proposed by the authors.
- Explain why the work matters.

If the user asks about methodology:
- Break down the method step by step.
- Explain the inputs, process, model/system/algorithm, and outputs.
- Mention equations, components, or architecture details only if they appear in the retrieved context.

If the user asks about results:
- Explain what was evaluated.
- Mention datasets, metrics, baselines, and performance numbers only if they are present in the retrieved context.
- Clearly explain what the results mean in simple language.

If the user asks about limitations:
- Use limitations stated or strongly supported by the retrieved context.
- If the paper does not clearly mention limitations, explain what can and cannot be concluded from the provided context without inventing unsupported criticism.

If the user asks for quiz questions, exam prep, or important questions:
- Create questions only from information supported by the retrieved context.
- Include answers if the user asks for them.
- Focus on concepts, methods, results, definitions, and comparisons from the paper.

If the user asks for flashcards, use exactly this format:

Q1: question
A1: answer

Q2: question
A2: answer

Q3: question
A3: answer

If the user request is vague, unclear, or too broad:

- Do not guess what format the user wants.
- Ask one short clarifying question.
- Offer clear options when helpful.
- Do not write a long answer until the user clarifies.

Examples of vague requests:
- "study mode"
- "help me study"
- "explain"
- "what should I know"
- "important"
- "quiz"
- "prepare me"

For a vague request like "study mode", respond like this:

What kind of study mode would you like?

1. Flashcards
2. Quiz questions
3. Simple summary notes
4. Key concepts
5. Exam-style questions

Once the user chooses, answer using only the retrieved paper context.

Do not add headings, bullet points, markdown symbols, or separators when creating flashcards.
Do not use ###.
Do not use ---.
Do not number flashcards in any format other than Q1/A1, Q2/A2, Q3/A3.

If the user asks for definitions:
- Define the term based on how it is used in the paper.
- If helpful, add a simple analogy or plain-English explanation, but do not introduce unsupported technical claims.

If the user asks for comparisons:
- Compare only the items discussed in the retrieved context.
- Use a simple table only if it makes the answer clearer.

If the user asks a question that is related to the paper but the retrieved context is incomplete:
- Answer as much as possible from the available context.
- Clearly say what part is supported by the paper.
- Mention what information is missing only when necessary.
- Do not say “the provided context does not contain enough information” unless the retrieved context is empty, unrelated, or truly does not support an answer.

If the retrieved context is empty or unrelated:
- Politely say that the retrieved paper sections do not contain enough relevant information to answer the question.
- Suggest asking about another section, uploading the full paper, or retrieving more context.

Do not:
- Hallucinate or guess unsupported details.
- Claim the paper says something unless it appears in the retrieved context.
- Use outside knowledge unless the user explicitly asks for general explanation beyond the paper.
- Overuse phrases like “the context says.”
- Use markdown headings like ###.
- Use horizontal separators like ---.
- Give citations that do not correspond to retrieved sources.

Tone:
Be clear, helpful, confident, and student-friendly. Explain like a knowledgeable research tutor who is helping the user understand the paper deeply.
"""

    user_prompt = f"""
Question:
{question}

Retrieved context:
{context}
"""

    return system_prompt, user_prompt


def generate_openai_answer(question: str, retrieved_chunks: list[dict]) -> dict | None:
    """
    Generate an answer using OpenAI if OPENAI_API_KEY exists.
    """

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return None

    if not retrieved_chunks:
        return {
            "answer": "I could not find relevant information in the uploaded document.",
            "llm_provider": "openai",
            "llm_model": OPENAI_MODEL,
        }

    client = OpenAI(api_key=api_key)

    system_prompt, user_prompt = build_prompts(
        question=question,
        retrieved_chunks=retrieved_chunks,
    )

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0.2,
        max_tokens=900,
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "llm_provider": "openai",
        "llm_model": OPENAI_MODEL,
    }


def generate_groq_answer(question: str, retrieved_chunks: list[dict]) -> dict | None:
    """
    Generate an answer using Groq if GROQ_API_KEY exists.
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return None

    if not retrieved_chunks:
        return {
            "answer": "I could not find relevant information in the uploaded document.",
            "llm_provider": "groq",
            "llm_model": GROQ_MODEL,
        }

    client = Groq(api_key=api_key)

    system_prompt, user_prompt = build_prompts(
        question=question,
        retrieved_chunks=retrieved_chunks,
    )

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0.2,
        max_tokens=900,
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "llm_provider": "groq",
        "llm_model": GROQ_MODEL,
    }


def generate_llm_answer(question: str, retrieved_chunks: list[dict]) -> dict | None:
    """
    Generate an LLM answer using available providers.

    Priority:
    1. OpenAI
    2. Groq
    3. None, so the app can use local fallback
    """

    openai_result = generate_openai_answer(
        question=question,
        retrieved_chunks=retrieved_chunks,
    )

    if openai_result:
        return openai_result

    groq_result = generate_groq_answer(
        question=question,
        retrieved_chunks=retrieved_chunks,
    )

    if groq_result:
        return groq_result

    return None