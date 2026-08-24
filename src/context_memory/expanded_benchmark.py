from __future__ import annotations

from .data import BenchmarkQuestion, QUESTIONS


# Ten lightweight query frames give us 300 deterministic questions without
# changing the underlying gold evidence. The goal is to test retrieval
# robustness to query phrasing before introducing semantic embeddings.
_FRAMES = (
    "In the current architecture, {query}",
    "Looking at the architecture today, {query}",
    "Based on the recorded decisions, {query}",
    "From the project history, {query}",
    "Can you determine {query}",
    "What does the project history tell us: {query}",
    "For the system as documented, {query}",
    "Considering the previous architecture changes, {query}",
    "According to the memory records, {query}",
    "What is the answer from the stored context: {query}",
)


def build_expanded_questions(base_questions: list[BenchmarkQuestion] = QUESTIONS) -> list[BenchmarkQuestion]:
    """Create 10 phrasing variants for every base question (300 total today)."""
    expanded: list[BenchmarkQuestion] = []
    for question in base_questions:
        for variant, frame in enumerate(_FRAMES, start=1):
            expanded.append(
                BenchmarkQuestion(
                    id=f"{question.id}-v{variant}",
                    category=question.category,
                    query=frame.format(query=question.query),
                    gold_evidence_ids=question.gold_evidence_ids,
                    complexity=question.complexity,
                )
            )
    return expanded


EXPANDED_QUESTIONS = build_expanded_questions()
