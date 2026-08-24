from dataclasses import dataclass
from time import perf_counter
import re

from .data import BenchmarkQuestion, MemoryChunk


@dataclass(frozen=True)
class RetrievalResult:
    strategy: str
    chunk_ids: tuple[str, ...]
    latency_ms: float


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2}


def score(query: str, chunk: MemoryChunk) -> int:
    return len(_tokens(query) & _tokens(chunk.text))


def full_context(question: BenchmarkQuestion, memories: list[MemoryChunk]) -> RetrievalResult:
    start = perf_counter()
    return RetrievalResult("full", tuple(m.id for m in memories), (perf_counter() - start) * 1000)


def fixed_top_k(question: BenchmarkQuestion, memories: list[MemoryChunk], k: int = 2) -> RetrievalResult:
    start = perf_counter()
    ranked = sorted(memories, key=lambda m: (-score(question.query, m), m.id))
    return RetrievalResult("fixed", tuple(m.id for m in ranked[:k]), (perf_counter() - start) * 1000)


def adaptive(question: BenchmarkQuestion, memories: list[MemoryChunk]) -> RetrievalResult:
    start = perf_counter()
    k = 2 if question.complexity == "simple" else 4
    ranked = sorted(memories, key=lambda m: (-score(question.query, m), m.id))
    return RetrievalResult("adaptive", tuple(m.id for m in ranked[:k]), (perf_counter() - start) * 1000)
