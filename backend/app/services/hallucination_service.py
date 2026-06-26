import re


def extract_source_labels(answer: str) -> list[str]:
    """
    Extract citation labels like [Source 1], [Source 2], etc. from an answer.
    """

    return re.findall(r"\[Source\s+\d+\]", answer or "")


def calculate_average_distance(retrieved_chunks: list[dict]) -> float | None:
    """
    Calculate average vector distance from retrieved chunks.
    Lower distance usually means stronger semantic relevance.
    """

    distances = []

    for chunk in retrieved_chunks:
        distance = chunk.get("distance")

        if isinstance(distance, (int, float)):
            distances.append(distance)

    if not distances:
        return None

    return sum(distances) / len(distances)


def detect_unsupported_language(answer: str) -> bool:
    """
    Check if the answer already admits that the paper context is insufficient.
    """

    answer_lower = (answer or "").lower()

    unsupported_phrases = [
        "could not find enough relevant information",
        "does not contain enough relevant information",
        "retrieved paper sections do not contain",
        "not enough relevant information",
        "cannot answer that confidently",
        "does not clearly mention",
        "not clearly provided",
        "not discussed",
    ]

    return any(phrase in answer_lower for phrase in unsupported_phrases)


def evaluate_grounding(
    answer: str,
    retrieved_chunks: list[dict],
    citations: list[str],
) -> dict:
    """
    Evaluate whether an answer appears grounded in retrieved paper context.

    This is a lightweight hallucination detection layer.
    It does not prove correctness, but it gives a useful grounding signal.
    """

    if not retrieved_chunks:
        return {
            "grounding_label": "Unsupported",
            "grounding_score": 0,
            "risk_level": "High",
            "explanation": "No relevant paper sections were retrieved for this question.",
            "source_count": 0,
            "citation_count": 0,
            "average_distance": None,
        }

    source_count = len(retrieved_chunks)
    citation_labels = extract_source_labels(answer)
    citation_count = len(citation_labels)
    average_distance = calculate_average_distance(retrieved_chunks)
    admits_missing_context = detect_unsupported_language(answer)

    score = 0

    # Retrieval exists
    if source_count >= 1:
        score += 25

    if source_count >= 3:
        score += 15

    # Citations exist
    if citation_count >= 1:
        score += 25

    if citation_count >= 2:
        score += 10

    # Fallback citations list exists
    if citations:
        score += 10

    # Distance-based signal, when available
    if average_distance is not None:
        if average_distance <= 0.8:
            score += 15
        elif average_distance <= 1.2:
            score += 8

    # If the answer admits missing support, lower confidence but do not punish as hallucination
    if admits_missing_context:
        score = min(score, 45)

    score = max(0, min(score, 100))

    if admits_missing_context:
        grounding_label = "Unsupported"
        risk_level = "Low hallucination risk"
        explanation = (
            "The answer correctly avoids making unsupported claims because the retrieved "
            "paper sections do not provide enough information."
        )
    elif score >= 75:
        grounding_label = "Strongly grounded"
        risk_level = "Low"
        explanation = "The answer uses retrieved paper sections and includes source citations."
    elif score >= 50:
        grounding_label = "Partially grounded"
        risk_level = "Medium"
        explanation = "The answer has some retrieved support, but citations or retrieval strength could be better."
    elif score >= 25:
        grounding_label = "Weakly grounded"
        risk_level = "Medium to High"
        explanation = "The answer has limited grounding and should be reviewed against the sources."
    else:
        grounding_label = "Unsupported"
        risk_level = "High"
        explanation = "The answer does not appear to be well supported by retrieved paper context."

    return {
        "grounding_label": grounding_label,
        "grounding_score": score,
        "risk_level": risk_level,
        "explanation": explanation,
        "source_count": source_count,
        "citation_count": citation_count,
        "average_distance": average_distance,
    }