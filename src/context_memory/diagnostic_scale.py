from __future__ import annotations

from dataclasses import asdict

from .data import BenchmarkQuestion, MemoryChunk
from .index_diagnostics import diagnose


def diagnostic_dicts(
    questions: list[BenchmarkQuestion],
    memories: list[MemoryChunk],
) -> list[dict]:
    """Return per-question index diagnostics for a scale experiment.

    This is intentionally separate from retrieval scoring: it tells us why an
    inverted index is selective (or not) without changing the retrieval path.
    """
    output: list[dict] = []
    for question in questions:
        result = diagnose(question.query, memories, question.gold_evidence_ids)
        item = asdict(result)
        item["terms"] = [asdict(term) for term in result.terms]
        output.append(item)
    return output
