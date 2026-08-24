from __future__ import annotations

from time import perf_counter

from .data import BenchmarkQuestion, MemoryChunk
from .judge import JudgeDecision, SufficiencyJudge
from .retrieval import InvertedIndex, RetrievalResult, _rank


class KeywordSufficiencyJudge:
    """Small deterministic judge used for tests and offline experiments."""

    def judge(self, question: BenchmarkQuestion, memories: list[MemoryChunk]) -> JudgeDecision:
        text = " ".join(m.text.lower() for m in memories)
        query_terms = {t for t in question.query.lower().split() if len(t) > 3}
        covered = sum(term in text for term in query_terms)
        ratio = covered / len(query_terms) if query_terms else 1.0
        return JudgeDecision(ratio >= 0.7, ratio, ())


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
    all_candidates = index.coarse_candidates(question.query) if index else memories
    retrieved: dict[str, MemoryChunk] = {}
    query = question.query
    rounds = 0
    candidates_scanned = len(all_candidates)

    while rounds < max_rounds:
        rounds += 1
        pool = index.coarse_candidates(query) if index else memories
        candidates_scanned += len(pool) if rounds > 1 else 0
        k = min(2 ** rounds, len(pool))
        ranked = _rank(question, pool, k)
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
                "llm_sufficient" if strategy.startswith("llm_") else "judge_sufficient",
            )

        if not decision.missing:
            query = f"{question.query} more context details rationale decision history"
        else:
            query = f"{question.query} {' '.join(decision.missing)}"

        if k >= len(pool) and not decision.missing:
            break

    return RetrievalResult(
        strategy,
        tuple(retrieved),
        (perf_counter() - start) * 1000,
        candidates_scanned,
        rounds,
        "max_rounds_reached",
    )


def llm_iterative(question: BenchmarkQuestion, memories: list[MemoryChunk], judge: SufficiencyJudge) -> RetrievalResult:
    return _run(question, memories, judge, "llm_iterative")


def indexed_llm_iterative(question: BenchmarkQuestion, index: InvertedIndex, judge: SufficiencyJudge) -> RetrievalResult:
    return _run(question, list(index.memories.values()), judge, "indexed_llm_iterative", indexed=True)
