from __future__ import annotations

from dataclasses import dataclass

from .data import MemoryChunk
from .selective_index import SelectiveMemoryIndex, terms


@dataclass(frozen=True)
class TermDiagnostic:
    term: str
    posting_count: int
    posting_ratio: float
    contains_gold: bool


@dataclass(frozen=True)
class IndexDiagnostic:
    query: str
    corpus_size: int
    terms: tuple[TermDiagnostic, ...]
    candidate_count: int
    candidate_ratio: float
    matched_terms: tuple[str, ...]
    gold_evidence_count: int
    gold_in_candidates: bool


def diagnose(
    query: str,
    memories: list[MemoryChunk],
    gold_evidence_ids: tuple[str, ...] = (),
) -> IndexDiagnostic:
    index = SelectiveMemoryIndex(memories)
    query_terms = sorted(terms(query))
    gold = set(gold_evidence_ids)
    diagnostics: list[TermDiagnostic] = []

    for term in query_terms:
        posting_ids = index.term_to_ids.get(term, set())
        diagnostics.append(
            TermDiagnostic(
                term=term,
                posting_count=len(posting_ids),
                posting_ratio=len(posting_ids) / max(len(memories), 1),
                contains_gold=bool(posting_ids & gold),
            )
        )

    candidates = index.lookup(query)
    candidate_ids = set(candidates.ids)

    return IndexDiagnostic(
        query=query,
        corpus_size=len(memories),
        terms=tuple(diagnostics),
        candidate_count=candidates.count,
        candidate_ratio=candidates.count / max(len(memories), 1),
        matched_terms=candidates.matched_terms,
        gold_evidence_count=len(gold),
        gold_in_candidates=gold.issubset(candidate_ids) if gold else True,
    )
