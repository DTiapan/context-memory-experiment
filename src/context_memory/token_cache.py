from __future__ import annotations

import re
from collections.abc import Iterable

from .data import MemoryChunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> frozenset[str]:
    """Normalize text once into the lexical representation used by retrieval."""
    return frozenset(t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 2)


class TokenCache:
    """Immutable-ish corpus/query token cache shared by retrieval strategies.

    The corpus is tokenized exactly once at construction. Query tokenization is
    cached as well because the same query is often inspected by several strategies.
    """

    def __init__(self, memories: Iterable[MemoryChunk]):
        self.memory_tokens: dict[str, frozenset[str]] = {
            memory.id: tokenize(memory.text) for memory in memories
        }
        self._query_tokens: dict[str, frozenset[str]] = {}

    def for_memory(self, memory: MemoryChunk) -> frozenset[str]:
        return self.memory_tokens[memory.id]

    def for_query(self, query: str) -> frozenset[str]:
        tokens = self._query_tokens.get(query)
        if tokens is None:
            tokens = tokenize(query)
            self._query_tokens[query] = tokens
        return tokens
