from dataclasses import dataclass
from time import perf_counter
import re

from .data import BenchmarkQuestion, MemoryChunk


@dataclass(frozen=True)
class RetrievalResult:
    strategy: str
    chunk_ids: tuple[str, ...]
    latency_ms: float
    candidate_count: int


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
