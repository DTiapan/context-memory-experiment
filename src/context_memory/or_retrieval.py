from __future__ import annotations

from dataclasses import dataclass
import math

from .data import MemoryChunk
from .selective_index import SelectiveMemoryIndex, terms
from .token_cache import TokenCache


@dataclass(frozen=True)
class RankedCandidate:
    memory: MemoryChunk
    score: float
    matched_terms: tuple[str, ...]


@dataclass(frozen=True)
class ORCandidateSet:
    candidates: tuple[RankedCandidate, ...]
    candidate_count: int
    candidate_ratio: float
    gold_evidence_in_candidates: bool


def retrieve_or(
    query: str,
    memories: list[MemoryChunk],
    *,
    top_k: int = 8,
    gold_evidence_ids: tuple[str, ...] = (),
    index: SelectiveMemoryIndex | None = None,
) -> ORCandidateSet:
    """Generate OR candidates using a prebuilt index when available."""
    index = index or SelectiveMemoryIndex(memories)
    query_terms = terms(query)
    corpus_size = max(len(memories), 1)

    scores: dict[str, float] = {}
    matched: dict[str, set[str]] = {}

    for term in query_terms:
        posting_ids = index.term_to_ids.get(term, set())
        if not posting_ids:
            continue
        idf = math.log((corpus_size + 1) / (len(posting_ids) + 1)) + 1.0
        for memory_id in posting_ids:
            scores[memory_id] = scores.get(memory_id, 0.0) + idf
            matched.setdefault(memory_id, set()).add(term)

    ranked = sorted(
        (
            RankedCandidate(index.memories[memory_id], score, tuple(sorted(matched[memory_id])))
            for memory_id, score in scores.items()
        ),
        key=lambda item: (-item.score, item.memory.id),
    )

    candidate_ids = set(scores)
    gold = set(gold_evidence_ids)
    return ORCandidateSet(
        candidates=tuple(ranked[:top_k]),
        candidate_count=len(candidate_ids),
        candidate_ratio=len(candidate_ids) / corpus_size,
        gold_evidence_in_candidates=gold.issubset(candidate_ids) if gold else True,
    )
