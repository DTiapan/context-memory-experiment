from dataclasses import dataclass

from .data import BenchmarkQuestion, MemoryChunk
from .retrieval import InvertedIndex, RetrievalResult


@dataclass(frozen=True)
class EvaluationRow:
    question_id: str
    category: str
    strategy: str
    retrieved_count: int
    candidate_count: int
    context_tokens: int
    recall: float
    all_evidence_retrieved: bool
    latency_ms: float
    retrieval_rounds: int


def estimate_tokens(chunks: list[MemoryChunk]) -> int:
    # Deliberately simple for MVP; replace with a real tokenizer later.
    return sum(max(1, len(chunk.text.split())) for chunk in chunks)


def evaluate_one(
    question: BenchmarkQuestion,
    result: RetrievalResult,
    memory_by_id: dict[str, MemoryChunk],
) -> EvaluationRow:
    gold = set(question.gold_evidence_ids)
    retrieved = set(result.chunk_ids)
    hits = len(gold & retrieved)
    recall = hits / len(gold) if gold else 1.0
    chunks = [memory_by_id[i] for i in result.chunk_ids]
    return EvaluationRow(
        question_id=question.id,
        category=question.category,
        strategy=result.strategy,
        retrieved_count=len(result.chunk_ids),
        candidate_count=result.candidate_count,
        context_tokens=estimate_tokens(chunks),
        recall=recall,
        all_evidence_retrieved=gold.issubset(retrieved),
        latency_ms=result.latency_ms,
        retrieval_rounds=result.retrieval_rounds,
    )


def _aggregate(items: list[EvaluationRow]) -> dict:
    return {
        "questions": len(items),
        "mean_recall": round(sum(r.recall for r in items) / len(items), 3),
        "full_evidence_rate": round(sum(r.all_evidence_retrieved for r in items) / len(items), 3),
        "mean_context_tokens": round(sum(r.context_tokens for r in items) / len(items), 1),
        "mean_retrieved_chunks": round(sum(r.retrieved_count for r in items) / len(items), 1),
        "mean_candidates_scanned": round(sum(r.candidate_count for r in items) / len(items), 1),
        "mean_retrieval_rounds": round(sum(r.retrieval_rounds for r in items) / len(items), 1),
        "mean_latency_ms": round(sum(r.latency_ms for r in items) / len(items), 4),
    }


def summarize(rows: list[EvaluationRow]) -> list[dict]:
    by_strategy: dict[str, list[EvaluationRow]] = {}
    for row in rows:
        by_strategy.setdefault(row.strategy, []).append(row)

    summaries = []
    for strategy, items in sorted(by_strategy.items()):
        summaries.append({"strategy": strategy, **_aggregate(items)})
    return summaries


def summarize_by_category(rows: list[EvaluationRow]) -> list[dict]:
    groups: dict[tuple[str, str], list[EvaluationRow]] = {}
    for row in rows:
        groups.setdefault((row.category, row.strategy), []).append(row)

    return [
        {"category": category, "strategy": strategy, **_aggregate(items)}
        for (category, strategy), items in sorted(groups.items())
    ]


def run_benchmark(questions: list[BenchmarkQuestion], memories: list[MemoryChunk]) -> tuple[list[EvaluationRow], list[dict], list[dict]]:
    from .retrieval import adaptive, fixed_top_k, full_context, indexed_adaptive, indexed_iterative, iterative

    index = InvertedIndex(memories)
    memory_by_id = {m.id: m for m in memories}
    rows: list[EvaluationRow] = []

    for question in questions:
        results = [
            full_context(question, memories),
            fixed_top_k(question, memories, k=2),
            adaptive(question, memories),
            indexed_adaptive(question, index),
            iterative(question, memories),
            indexed_iterative(question, index),
        ]
        for result in results:
            rows.append(evaluate_one(question, result, memory_by_id))

    return rows, summarize(rows), summarize_by_category(rows)
