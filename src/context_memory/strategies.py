from collections.abc import Callable

from .data import BenchmarkQuestion, MemoryChunk
from .retrieval import RetrievalResult, adaptive, fixed_top_k, full_context

Strategy = Callable[[BenchmarkQuestion, list[MemoryChunk]], RetrievalResult]


def fixed_strategy(question: BenchmarkQuestion, memories: list[MemoryChunk]) -> RetrievalResult:
    return fixed_top_k(question, memories, k=2)


STRATEGIES: dict[str, Strategy] = {
    "full": full_context,
    "fixed": fixed_strategy,
    "adaptive": adaptive,
}
