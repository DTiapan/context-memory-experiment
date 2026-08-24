from dataclasses import dataclass

from .data import BenchmarkQuestion, MemoryChunk
from .retrieval import (
    InvertedIndex,
    adaptive,
    fixed_top_k,
    full_context,
    heuristic_iterative,
    indexed_adaptive,
    indexed_heuristic_iterative,
    indexed_iterative,
    iterative,
)


@dataclass(frozen=True)
class EvaluationRowV3:
    question_id: str
    category: str
    strategy: str
    retrieved_count: int
    candidate_count: int
    context_tokens: int
    recall: float
    all_evidence_retrieved: bool
    retrieval_rounds: int
    stopping_reason: str


def _tokens(chunks: list[MemoryChunk]) -> int:
    return sum(max(1, len(chunk.text.split())) for chunk in chunks)


def evaluate(question: BenchmarkQuestion, result, memory_by_id: dict[str, MemoryChunk]) -> EvaluationRowV3:
    gold = set(question.gold_evidence_ids)
    retrieved = set(result.chunk_ids)
    hits = len(gold & retrieved)
    chunks = [memory_by_id[i] for i in result.chunk_ids]
    return EvaluationRowV3(
        question.id,
        question.category,
        result.strategy,
        len(result.chunk_ids),
        result.candidate_count,
        _tokens(chunks),
        hits / len(gold) if gold else 1.0,
        gold.issubset(retrieved),
        result.retrieval_rounds,
        result.stopping_reason,
    )


def _aggregate(rows: list[EvaluationRowV3]) -> dict:
    return {
        "questions": len(rows),
        "mean_recall": round(sum(r.recall for r in rows) / len(rows), 3),
        "full_evidence_rate": round(sum(r.all_evidence_retrieved for r in rows) / len(rows), 3),
        "mean_context_tokens": round(sum(r.context_tokens for r in rows) / len(rows), 1),
        "mean_retrieved_chunks": round(sum(r.retrieved_count for r in rows) / len(rows), 1),
        "mean_candidates_scanned": round(sum(r.candidate_count for r in rows) / len(rows), 1),
        "mean_retrieval_rounds": round(sum(r.retrieval_rounds for r in rows) / len(rows), 1),
    }


def run_benchmark_v3(questions: list[BenchmarkQuestion], memories: list[MemoryChunk]):
    index = InvertedIndex(memories)
    memory_by_id = {m.id: m for m in memories}
    rows: list[EvaluationRowV3] = []
    for question in questions:
        results = [
            full_context(question, memories),
            fixed_top_k(question, memories),
            adaptive(question, memories),
            indexed_adaptive(question, index),
            iterative(question, memories),
            indexed_iterative(question, index),
            heuristic_iterative(question, memories),
            indexed_heuristic_iterative(question, index),
        ]
        rows.extend(evaluate(question, result, memory_by_id) for result in results)

    by_strategy = {}
    by_category = {}
    by_reason = {}
    for row in rows:
        by_strategy.setdefault(row.strategy, []).append(row)
        by_category.setdefault((row.category, row.strategy), []).append(row)
        by_reason.setdefault((row.strategy, row.stopping_reason), []).append(row)

    return (
        rows,
        [{"strategy": k, **_aggregate(v)} for k, v in sorted(by_strategy.items())],
        [{"category": k[0], "strategy": k[1], **_aggregate(v)} for k, v in sorted(by_category.items())],
        [{"strategy": k[0], "stopping_reason": k[1], **_aggregate(v)} for k, v in sorted(by_reason.items())],
    )
