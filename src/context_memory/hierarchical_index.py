from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .data import MemoryChunk
from .selective_index import terms


@dataclass(frozen=True)
class HierarchicalCandidateSet:
    group_ids: tuple[str, ...]
    memory_ids: tuple[str, ...]
    matched_terms: tuple[str, ...]

    @property
    def count(self) -> int:
        return len(self.memory_ids)


class HierarchicalMemoryIndex:
    """Two-stage lexical index: query terms select groups, then group members.

    A group is a topic bucket. Entity names are also indexed against the group,
    so queries can jump to a small semantic-ish partition before ranking
    individual memories. This MVP deliberately avoids embeddings and LLMs.
    """

    def __init__(self, memories: list[MemoryChunk]):
        self.memories = {m.id: m for m in memories}
        self.group_to_ids: dict[str, set[str]] = defaultdict(set)
        self.term_to_groups: dict[str, set[str]] = defaultdict(set)

        for memory in memories:
            group_id = memory.topic
            self.group_to_ids[group_id].add(memory.id)
            group_terms = terms(f"{memory.topic} {' '.join(memory.entities)}")
            for term in group_terms:
                self.term_to_groups[term].add(group_id)

    def lookup(self, query: str, *, max_groups: int = 8) -> HierarchicalCandidateSet:
        query_terms = terms(query)
        matched_groups: set[str] = set()
        matched_terms: list[str] = []

        # OR at the group level is intentional: missing one query term should
        # not eliminate an otherwise relevant topic bucket.
        for term in sorted(query_terms):
            groups = self.term_to_groups.get(term)
            if groups:
                matched_groups.update(groups)
                matched_terms.append(term)

        if not matched_groups:
            matched_groups = set(self.group_to_ids)

        # Prefer groups matched by more query terms, then stable ordering.
        scored_groups = sorted(
            matched_groups,
            key=lambda group: (-sum(
                1 for term in query_terms if group in self.term_to_groups.get(term, set())
            ), group),
        )[:max_groups]

        memory_ids = sorted(
            memory_id
            for group in scored_groups
            for memory_id in self.group_to_ids[group]
        )
        return HierarchicalCandidateSet(
            tuple(scored_groups),
            tuple(memory_ids),
            tuple(matched_terms),
        )
