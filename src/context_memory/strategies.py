from collections.abc import Callable

from .data import BenchmarkQuestion, MemoryChunk
from .retrieval import (
    RetrievalResult,
    adaptive,
    fixed_top_k,
    full_context,
    indexed_adaptive,
    indexed_iterative,
    iterative,
    InvertedIndex,
)

Strategy = Callable[[BenchmarkQuestion, list[MemoryChunk]], RetrievalResult]


def fixed_strategy(question: BenchmarkQuestion, memories: list[MemoryChunk]) -> RetrievalResult:
    return fixed_top_k(question, memories, k=2)


STRATEGIES: dict[str, Strategy] = {
    "full": full_context,
    "fixed": fixed_strategy,
    "adaptive": adaptive,
    "iterative": iterative,
}


def indexed_strategies(memories: list[MemoryChunk]) -> dict[str, Strategy]:
    index = InvertedIndex(memories)

    return {
        "indexed_adaptive": lambda question, _: indexed_adaptive(question, index),
        "indexed_iterative": lambda question, _: indexed_iterative(question, index),
    }
