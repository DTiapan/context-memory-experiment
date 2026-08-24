from __future__ import annotations

from dataclasses import dataclass

from .data import BenchmarkQuestion, MemoryChunk
from .selective_index import SelectiveMemoryIndex


@dataclass(frozen=True)
class SelectiveRetrievalResult:
    memories: tuple[MemoryChunk, ...]
    candidate_count: int
    candidate_ratio: float
    matched_terms: tuple[str, ...]


def retrieve(question: BenchmarkQuestion, memories: list[MemoryChunk], top_k: int = 8) -> SelectiveRetrievalResult:
    index = SelectiveMemoryIndex(memories)
    candidates = index.lookup(question.query)
    candidate_memories = [index.memories[memory_id] for memory_id in candidates.ids]

    # Preserve the existing lexical scoring semantics while restricting the
    # expensive ranking stage to the selective candidate set.
    query_terms = set(question.query.lower().split())
    scored = sorted(
        candidate_memories,
        key=lambda memory: len(query_terms & set(memory.text.lower().split())),
        reverse=True,
    )
    selected = tuple(scored[:top_k])
    return SelectiveRetrievalResult(
        memories=selected,
        candidate_count=candidates.count,
        candidate_ratio=candidates.count / max(len(memories), 1),
        matched_terms=candidates.matched_terms,
    )
