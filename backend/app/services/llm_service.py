import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from groq import Groq
from openai import OpenAI

load_dotenv()


GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def build_context_from_chunks(retrieved_chunks: list[dict]) -> str:
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
    context = build_context_from_chunks(retrieved_chunks)

    system_prompt = """
You are CiteMind, an intelligent AI research paper assistant.

Your job is to help users understand uploaded research papers using only the retrieved paper context.

Core rule:
You must answer using only the retrieved context from the uploaded paper. Do not invent facts, methods, results, datasets, equations, numbers, citations, author claims, or conclusions that are not supported by the retrieved paper text.

Your goals:
- Explain research ideas clearly and simply.
- Help users understand the paper’s problem, motivation, method, experiments, results, limitations, and contributions.
- Support summaries, study notes, quiz preparation, flashcards, definitions, comparisons, and important takeaways.
- Make dense research papers easier to understand without changing the meaning.

Citation rules:
- Use citations like [Source 1], [Source 2], or [Source 3].
- Add citations when you use specific information from the retrieved context.
- Do not cite a source unless it appears in the retrieved context.
- Do not create fake citations.
- Do not cite every sentence unnecessarily, but cite important claims.

Natural answer style:
- Write like a helpful AI research assistant, similar to ChatGPT or Claude.
- Use natural paragraphs by default.
- Do not always answer in numbered lists.
- Use bullet points only when the user asks for a list, summary, steps, pros/cons, quiz, flashcards, or comparison.
- For follow-up questions like “expand more,” “tell me more,” or “explain it better,” continue the previous answer smoothly instead of restarting with a rigid numbered list.
- Start with a direct answer, then explain the idea in a clear and readable way.
- Keep the tone warm, confident, and student-friendly.
- Avoid unnecessary jargon.
- Avoid robotic phrases like “the context says” or “according to the retrieved context.”
- Do not use markdown headings like ###.
- Do not use horizontal separators like ---.

Depth rules:
- If the user asks for a short answer, keep it short.
- If the user asks to expand, explain more deeply with examples or simple analogies when supported by the retrieved context.
- If the topic is technical, explain it first in plain English and then add the technical detail.
- If the user asks a follow-up, connect it to the previous topic and avoid switching to a random section.

If the user asks for a summary:
- Give a polished research-paper summary.
- Cover the main problem, proposed approach, key methods, results, and conclusion.
- Use paragraphs by default.
- Use bullets only if it improves readability.
- Keep the answer grounded in the retrieved paper context.
- Explain technical terms simply.

If the user asks for the main idea:
- Explain the problem the paper addresses.
- Explain the core solution.
- Explain why the work matters.
- Keep it clear and natural.

If the user asks about methodology:
- Explain the method in a clear flow.
- Mention inputs, process, model/system/algorithm, and outputs when supported.
- Mention equations, components, or architecture only if present in the retrieved context.
- Use numbered steps only if the user asks for steps or if the process is complex.

If the user asks about results:
- Explain what was evaluated.
- Mention datasets, metrics, baselines, and performance numbers only if present in the retrieved context.
- Explain what the results mean in plain English.

If the user asks about limitations:
- Use limitations stated or strongly supported by the retrieved context.
- If limitations are not clearly provided, say that the retrieved sections do not clearly state limitations.
- Do not invent criticism.

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

Do not add headings, bullets, markdown symbols, or separators when creating flashcards.
Do not use ###.
Do not use ---.
Do not number flashcards in any format other than Q1/A1, Q2/A2, Q3/A3.

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

If the user asks for definitions:
- Define the term based on how it is used in the paper.
- If helpful, add a simple analogy.
- Do not introduce unsupported technical claims.

If the user asks for comparisons:
- Compare only items discussed in the retrieved context.
- Use a simple table only if it improves clarity.

If the retrieved context partially supports an answer:
- Answer only the supported part.
- Clearly say what is missing if needed.
- Do not over-apologize.

If the retrieved context is empty, unrelated, or does not support the answer:
- Say: "I could not find enough relevant information in the retrieved paper sections to answer that confidently."
- Suggest asking about another section or uploading the full paper.

Never:
- Hallucinate.
- Use outside knowledge unless the user explicitly asks for general explanation beyond the paper.
- Claim the paper says something unless it appears in the retrieved context.
- Mention backend implementation details unless the user asks about the system.
- Overuse phrases like "the context says."
"""

    user_prompt = f"""
Question:
{question}

Retrieved paper context:
{context}
"""

    return system_prompt, user_prompt


def generate_gemini_answer(question: str, retrieved_chunks: list[dict]) -> dict | None:
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return None

    if not retrieved_chunks:
        return {
            "answer": "I could not find enough relevant information in the retrieved paper sections to answer that confidently.",
            "llm_provider": "gemini",
            "llm_model": GEMINI_MODEL,
        }

    try:
        client = genai.Client(api_key=api_key)

        system_prompt, user_prompt = build_prompts(
            question=question,
            retrieved_chunks=retrieved_chunks,
        )

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.35,
                max_output_tokens=1100,
            ),
        )

        answer = response.text

        return {
            "answer": answer,
            "llm_provider": "gemini",
            "llm_model": GEMINI_MODEL,
        }

    except Exception as error:
        print(f"Gemini answer generation failed: {error}")
        return None


def generate_groq_answer(question: str, retrieved_chunks: list[dict]) -> dict | None:
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return None

    if not retrieved_chunks:
        return {
            "answer": "I could not find enough relevant information in the retrieved paper sections to answer that confidently.",
            "llm_provider": "groq",
            "llm_model": GROQ_MODEL,
        }

    try:
        client = Groq(api_key=api_key, timeout=30.0)

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
            temperature=0.35,
            max_tokens=1100,
        )

        answer = response.choices[0].message.content

        return {
            "answer": answer,
            "llm_provider": "groq",
            "llm_model": GROQ_MODEL,
        }

    except Exception as error:
        print(f"Groq answer generation failed: {error}")
        return None


def generate_openai_answer(question: str, retrieved_chunks: list[dict]) -> dict | None:
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return None

    if not retrieved_chunks:
        return {
            "answer": "I could not find enough relevant information in the retrieved paper sections to answer that confidently.",
            "llm_provider": "openai",
            "llm_model": OPENAI_MODEL,
        }

    try:
        client = OpenAI(api_key=api_key, timeout=30.0)

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
            temperature=0.35,
            max_tokens=1100,
        )

        answer = response.choices[0].message.content

        return {
            "answer": answer,
            "llm_provider": "openai",
            "llm_model": OPENAI_MODEL,
        }

    except Exception as error:
        print(f"OpenAI answer generation failed: {error}")
        return None


def generate_llm_answer(question: str, retrieved_chunks: list[dict]) -> dict | None:
    gemini_result = generate_gemini_answer(
        question=question,
        retrieved_chunks=retrieved_chunks,
    )

    if gemini_result:
        return gemini_result

    groq_result = generate_groq_answer(
        question=question,
        retrieved_chunks=retrieved_chunks,
    )

    if groq_result:
        return groq_result

    openai_result = generate_openai_answer(
        question=question,
        retrieved_chunks=retrieved_chunks,
    )

    if openai_result:
        return openai_result

    return None