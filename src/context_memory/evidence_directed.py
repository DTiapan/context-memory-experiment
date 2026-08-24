from __future__ import annotations

from time import perf_counter

from .data import BenchmarkQuestion, MemoryChunk
from .judge import JudgeDecision, SufficiencyJudge
from .retrieval import InvertedIndex, RetrievalResult, score


class KeywordSufficiencyJudge:
    """Small deterministic judge used for tests and offline experiments."""

    def judge(self, question: BenchmarkQuestion, memories: list[MemoryChunk]) -> JudgeDecision:
        text = " ".join(m.text.lower() for m in memories)
        query_terms = {t for t in question.query.lower().split() if len(t) > 3}
        covered = sum(term in text for term in query_terms)
        ratio = covered / len(query_terms) if query_terms else 1.0
        return JudgeDecision(ratio >= 0.7, ratio, ())


def _rank_query(query: str, memories: list[MemoryChunk], k: int) -> list[MemoryChunk]:
    return sorted(memories, key=lambda m: (-score(query, m), m.id))[:k]


def _run(
    question: BenchmarkQuestion,
    memories: list[MemoryChunk],
    judge: SufficiencyJudge,
    strategy: str,
    max_rounds: int = 5,
    indexed: bool = False,
) -> RetrievalResult:
    start = perf_counter()
    index = InvertedIndex(memories) if indexed else None
    retrieved: dict[str, MemoryChunk] = {}
    query = question.query
    rounds = 0
    candidates_scanned = 0

    while rounds < max_rounds:
        rounds += 1
        pool = index.coarse_candidates(query) if index else memories
        candidates_scanned += len(pool)
        if not pool:
            break

        k = min(2 ** rounds, len(pool))
        ranked = _rank_query(query, pool, k)
        for chunk in ranked:
            retrieved[chunk.id] = chunk

        decision = judge.judge(question, list(retrieved.values()))
        if decision.sufficient:
            return RetrievalResult(
                strategy,
                tuple(retrieved),
                (perf_counter() - start) * 1000,
                candidates_scanned,
                rounds,
                "judge_sufficient",
            )

        query = f"{question.query} {' '.join(decision.missing)}".strip()
        if not decision.missing and k >= len(pool):
            break

    return RetrievalResult(
        strategy,
        tuple(retrieved),
        (perf_counter() - start) * 1000,
        candidates_scanned,
        rounds,
        "max_rounds_reached",
    )


def llm_iterative(
    question: BenchmarkQuestion,
    memories: list[MemoryChunk],
    judge: SufficiencyJudge,
    max_rounds: int = 5,
) -> RetrievalResult:
    return _run(question, memories, judge, "llm_iterative", max_rounds=max_rounds)


def indexed_llm_iterative(
    question: BenchmarkQuestion,
    index: InvertedIndex,
    judge: SufficiencyJudge,
    max_rounds: int = 5,
) -> RetrievalResult:
    return _run(
        question,
        list(index.memories.values()),
        judge,
        "indexed_llm_iterative",
        max_rounds=max_rounds,
        indexed=True,
    )
