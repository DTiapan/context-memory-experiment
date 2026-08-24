from dataclasses import dataclass
from time import perf_counter
import re

from .data import BenchmarkQuestion, MemoryChunk
from .sufficiency import heuristic_sufficiency


@dataclass(frozen=True)
class RetrievalResult:
    strategy: str
    chunk_ids: tuple[str, ...]
    latency_ms: float
    candidate_count: int
    retrieval_rounds: int = 1
    stopping_reason: str = "fixed_k"


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2}


def score(query: str, chunk: MemoryChunk) -> int:
    return len(_tokens(query) & _tokens(chunk.text))


class InvertedIndex:
    """Tiny inverted index used to test the index -> deeper lookup idea."""

    def __init__(self, memories: list[MemoryChunk]):
        self.memories = {m.id: m for m in memories}
        self.token_index: dict[str, set[str]] = {}
        self.entity_index: dict[str, set[str]] = {}
        for memory in memories:
            for token in _tokens(memory.text):
                self.token_index.setdefault(token, set()).add(memory.id)
            for entity in memory.entities:
                self.entity_index.setdefault(entity.lower(), set()).add(memory.id)

    def coarse_candidates(self, query: str) -> list[MemoryChunk]:
        query_tokens = _tokens(query)
        ids: set[str] = set()
        for token in query_tokens:
            ids.update(self.token_index.get(token, set()))
        for entity, entity_ids in self.entity_index.items():
            if entity in query_tokens:
                ids.update(entity_ids)
        return [self.memories[i] for i in sorted(ids)]


def _rank(question: BenchmarkQuestion, memories: list[MemoryChunk], k: int) -> list[MemoryChunk]:
    return sorted(memories, key=lambda m: (-score(question.query, m), m.id))[:k]


def full_context(question: BenchmarkQuestion, memories: list[MemoryChunk]) -> RetrievalResult:
    start = perf_counter()
    return RetrievalResult("full", tuple(m.id for m in memories), (perf_counter() - start) * 1000, len(memories))


def fixed_top_k(question: BenchmarkQuestion, memories: list[MemoryChunk], k: int = 2) -> RetrievalResult:
    start = perf_counter()
    ranked = _rank(question, memories, k)
    return RetrievalResult("fixed", tuple(m.id for m in ranked), (perf_counter() - start) * 1000, len(memories))


def adaptive(question: BenchmarkQuestion, memories: list[MemoryChunk]) -> RetrievalResult:
    start = perf_counter()
    k = 2 if question.complexity == "simple" else 4
    ranked = _rank(question, memories, k)
    return RetrievalResult("adaptive", tuple(m.id for m in ranked), (perf_counter() - start) * 1000, len(memories))


def indexed_adaptive(question: BenchmarkQuestion, index: InvertedIndex) -> RetrievalResult:
    start = perf_counter()
    candidates = index.coarse_candidates(question.query)
    k = 2 if question.complexity == "simple" else 4
    ranked = _rank(question, candidates, k)
    return RetrievalResult(
        "indexed_adaptive",
        tuple(m.id for m in ranked),
        (perf_counter() - start) * 1000,
        len(candidates),
    )


def _evidence_complete(question: BenchmarkQuestion, ranked: list[MemoryChunk]) -> bool:
    """Oracle used only by the controlled benchmark to test retrieval depth."""
    return set(question.gold_evidence_ids).issubset({m.id for m in ranked})


def _progressive_rank(
    question: BenchmarkQuestion,
    memories: list[MemoryChunk],
    start_k: int = 2,
    max_k: int | None = None,
) -> tuple[list[MemoryChunk], int, int]:
    """Retrieve in expanding batches until the benchmark evidence is complete."""
    ordered = sorted(memories, key=lambda m: (-score(question.query, m), m.id))
    max_k = max_k or len(ordered)
    k = min(start_k, max_k)
    rounds = 0
    while True:
        rounds += 1
        ranked = ordered[:k]
        if _evidence_complete(question, ranked) or k >= max_k:
            return ranked, k, rounds
        k = min(k * 2, max_k)


def _heuristic_progressive_rank(
    question: BenchmarkQuestion,
    memories: list[MemoryChunk],
    start_k: int = 2,
    max_k: int | None = None,
) -> tuple[list[MemoryChunk], int, int, str]:
    """Progressive retrieval using an evidence-sufficiency heuristic, no gold labels."""
    ordered = sorted(memories, key=lambda m: (-score(question.query, m), m.id))
    max_k = max_k or len(ordered)
    k = min(start_k, max_k)
    rounds = 0
    while True:
        rounds += 1
        ranked = ordered[:k]
        decision = heuristic_sufficiency(question, ranked)
        if decision.sufficient or k >= max_k:
            reason = "heuristic_sufficient" if decision.sufficient else "max_k_reached"
            return ranked, k, rounds, reason
        k = min(k * 2, max_k)


def iterative(question: BenchmarkQuestion, memories: list[MemoryChunk]) -> RetrievalResult:
    start = perf_counter()
    ranked, scanned, rounds = _progressive_rank(question, memories)
    return RetrievalResult(
        "iterative",
        tuple(m.id for m in ranked),
        (perf_counter() - start) * 1000,
        scanned,
        rounds,
        "oracle_evidence_complete",
    )


def indexed_iterative(question: BenchmarkQuestion, index: InvertedIndex) -> RetrievalResult:
    start = perf_counter()
    candidates = index.coarse_candidates(question.query)
    ranked, scanned, rounds = _progressive_rank(question, candidates)
    return RetrievalResult(
        "indexed_iterative",
        tuple(m.id for m in ranked),
        (perf_counter() - start) * 1000,
        scanned,
        rounds,
        "oracle_evidence_complete",
    )


def heuristic_iterative(question: BenchmarkQuestion, memories: list[MemoryChunk]) -> RetrievalResult:
    start = perf_counter()
    ranked, scanned, rounds, reason = _heuristic_progressive_rank(question, memories)
    return RetrievalResult(
        "heuristic_iterative",
        tuple(m.id for m in ranked),
        (perf_counter() - start) * 1000,
        scanned,
        rounds,
        reason,
    )


def indexed_heuristic_iterative(question: BenchmarkQuestion, index: InvertedIndex) -> RetrievalResult:
    start = perf_counter()
    candidates = index.coarse_candidates(question.query)
    ranked, scanned, rounds, reason = _heuristic_progressive_rank(question, candidates)
    return RetrievalResult(
        "indexed_heuristic_iterative",
        tuple(m.id for m in ranked),
        (perf_counter() - start) * 1000,
        scanned,
        rounds,
        reason,
    )
