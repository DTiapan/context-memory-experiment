from dataclasses import asdict, dataclass

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


def estimate_tokens(chunks: list[MemoryChunk]) -> int:
    # Deliberately simple for MVP 1; replace with a real tokenizer later.
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
    )


def summarize(rows: list[EvaluationRow]) -> list[dict]:
    by_strategy: dict[str, list[EvaluationRow]] = {}
    for row in rows:
        by_strategy.setdefault(row.strategy, []).append(row)

    summaries = []
    for strategy, items in sorted(by_strategy.items()):
        summaries.append(
            {
                "strategy": strategy,
                "questions": len(items),
                "mean_recall": round(sum(r.recall for r in items) / len(items), 3),
                "full_evidence_rate": round(sum(r.all_evidence_retrieved for r in items) / len(items), 3),
                "mean_context_tokens": round(sum(r.context_tokens for r in items) / len(items), 1),
                "mean_retrieved_chunks": round(sum(r.retrieved_count for r in items) / len(items), 1),
                "mean_candidates_scanned": round(sum(r.candidate_count for r in items) / len(items), 1),
                "mean_candidate_reduction": round(
                    1 - (sum(r.candidate_count for r in items) / len(items)) / max(1, sum(r.candidate_count for r in items) / len(items)), 3
                ) if strategy == "full" else None,
                "mean_latency_ms": round(sum(r.latency_ms for r in items) / len(items), 4),
            }
        )
    return summaries


def run_benchmark(questions: list[BenchmarkQuestion], memories: list[MemoryChunk]) -> tuple[list[EvaluationRow], list[dict]]:
    from .retrieval import adaptive, fixed_top_k, full_context, indexed_adaptive

    index = InvertedIndex(memories)
    memory_by_id = {m.id: m for m in memories}
    rows: list[EvaluationRow] = []

    for question in questions:
        results = [
            full_context(question, memories),
            fixed_top_k(question, memories, k=2),
            adaptive(question, memories),
            indexed_adaptive(question, index),
        ]
        for result in results:
            rows.append(evaluate_one(question, result, memory_by_id))

    return rows, summarize(rows)
