from __future__ import annotations

from collections import defaultdict
import re
from dataclasses import dataclass

from .data import MemoryChunk

_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with",
    "why", "what", "when", "how", "did", "we", "our", "was", "were", "is", "are",
    "from", "about", "into", "by", "it", "this", "that", "after", "before",
}


def terms(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text)
        if token.lower() not in _STOPWORDS
    }


@dataclass(frozen=True)
class CandidateSet:
    ids: tuple[str, ...]
    matched_terms: tuple[str, ...]

    @property
    def count(self) -> int:
        return len(self.ids)


class SelectiveMemoryIndex:
    """Multi-term inverted index with intersection-based candidate selection.

    The index is intentionally lexical for MVP-5. The experiment is testing
    selectivity and candidate reduction before introducing semantic retrieval.
    """

    def __init__(self, memories: list[MemoryChunk]):
        self.memories = {m.id: m for m in memories}
        self.term_to_ids: dict[str, set[str]] = defaultdict(set)
        for memory in memories:
            for term in terms(f"{memory.text} {memory.id}"):
                self.term_to_ids[term].add(memory.id)

    def lookup(self, query: str, *, min_terms: int = 1) -> CandidateSet:
        query_terms = terms(query)
        postings = [(term, self.term_to_ids.get(term, set())) for term in query_terms]
        postings = [(term, ids) for term, ids in postings if ids]
        if not postings:
            return CandidateSet(tuple(self.memories), ())

        # Start with the rarest term, then add more terms only when they
        # preserve candidates. This gives us intersection-based selectivity
        # without making every query fail when one term is absent from the
        # relevant memory.
        postings.sort(key=lambda item: len(item[1]))
        candidate_ids = set(postings[0][1])
        matched = [postings[0][0]]

        for term, ids in postings[1:]:
            intersection = candidate_ids & ids
            if intersection:
                candidate_ids = intersection
                matched.append(term)

        return CandidateSet(tuple(sorted(candidate_ids)), tuple(matched))

    def selectivity(self, query: str) -> float:
        result = self.lookup(query)
        return result.count / max(len(self.memories), 1)
