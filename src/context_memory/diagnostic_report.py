from __future__ import annotations

from .data import BenchmarkQuestion, MemoryChunk
from .index_diagnostics import IndexDiagnostic, diagnose


def format_diagnostic(question: BenchmarkQuestion, memories: list[MemoryChunk]) -> str:
    result: IndexDiagnostic = diagnose(question.query, memories, question.gold_evidence_ids)
    lines = [
        f"question={question.id}",
        f"query={question.query}",
        f"corpus_size={result.corpus_size}",
        f"candidate_count={result.candidate_count}",
        f"candidate_ratio={result.candidate_ratio:.6f}",
        f"matched_terms={','.join(result.matched_terms)}",
        f"gold_evidence_count={result.gold_evidence_count}",
        f"gold_in_candidates={result.gold_in_candidates}",
        "terms:",
    ]
    for item in result.terms:
        lines.append(
            f"  {item.term}: posting_count={item.posting_count} "
            f"posting_ratio={item.posting_ratio:.6f} contains_gold={item.contains_gold}"
        )
    return "\n".join(lines)
