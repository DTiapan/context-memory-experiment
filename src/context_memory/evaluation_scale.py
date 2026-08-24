from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

from .data import BenchmarkQuestion, MemoryChunk
from .retrieval import InvertedIndex, fixed_top_k, indexed_iterative, iterative
from .or_retrieval import retrieve_or
from .hierarchical_index import HierarchicalMemoryIndex


@dataclass(frozen=True)
class ScaleRow:
    corpus_size: int
    question_id: str
    category: str
    strategy: str
    retrieved_count: int
    candidates_scanned: int
    context_tokens: int
    recall: float
    all_evidence_retrieved: bool
    retrieval_rounds: int
    latency_ms: float


def _tokens(memories: list[MemoryChunk]) -> int:
    return sum(max(1, len(m.text.split())) for m in memories)


def _row(size: int, question: BenchmarkQuestion, result, by_id) -> ScaleRow:
    gold = set(question.gold_evidence_ids)
    retrieved = set(result.chunk_ids)
    chunks = [by_id[i] for i in result.chunk_ids]
    return ScaleRow(
        size,
        question.id,
        question.category,
        result.strategy,
        len(result.chunk_ids),
        result.candidate_count,
        _tokens(chunks),
        len(gold & retrieved) / len(gold),
        gold.issubset(retrieved),
        result.retrieval_rounds,
        result.latency_ms,
    )


def _or_row(size: int, question: BenchmarkQuestion, memories: list[MemoryChunk], by_id) -> ScaleRow:
    start = perf_counter()
    result = retrieve_or(
        question.query,
        memories,
        top_k=8,
        gold_evidence_ids=question.gold_evidence_ids,
    )
    latency_ms = (perf_counter() - start) * 1000
    retrieved_ids = [candidate.memory.id for candidate in result.candidates]
    gold = set(question.gold_evidence_ids)
    retrieved = set(retrieved_ids)
    chunks = [by_id[i] for i in retrieved_ids]
    return ScaleRow(
        size,
        question.id,
        question.category,
        "or_ranked",
        len(retrieved_ids),
        result.candidate_count,
        _tokens(chunks),
        len(gold & retrieved) / len(gold),
        gold.issubset(retrieved),
        1,
        latency_ms,
    )


def _hierarchical_row(size: int, question: BenchmarkQuestion, index: HierarchicalMemoryIndex, by_id) -> ScaleRow:
    start = perf_counter()
    candidate_set = index.lookup(question.query, max_groups=4)
    candidates = [index.memories[memory_id] for memory_id in candidate_set.memory_ids]

    # The group lookup is the narrowing stage. Ranking is deliberately cheap
    # lexical overlap so this experiment isolates hierarchy from semantics.
    query_terms = set(question.query.lower().split())
    scored = sorted(
        candidates,
        key=lambda memory: len(query_terms & set(memory.text.lower().split())),
        reverse=True,
    )[:8]
    latency_ms = (perf_counter() - start) * 1000

    gold = set(question.gold_evidence_ids)
    retrieved = {memory.id for memory in scored}
    return ScaleRow(
        size,
        question.id,
        question.category,
        "hierarchical",
        len(scored),
        candidate_set.count,
        _tokens(scored),
        len(gold & retrieved) / len(gold),
        gold.issubset(retrieved),
        2,
        latency_ms,
    )


def run_scale_point(size: int, questions: list[BenchmarkQuestion], memories: list[MemoryChunk]) -> list[ScaleRow]:
    index = InvertedIndex(memories)
    hierarchical_index = HierarchicalMemoryIndex(memories)
    by_id = {m.id: m for m in memories}
    rows: list[ScaleRow] = []
    for question in questions:
        # Sequential baselines scan the entire corpus. Indexed iterative uses
        # the inverted index before the same progressive ranking procedure.
        rows.append(_row(size, question, fixed_top_k(question, memories), by_id))
        rows.append(_row(size, question, iterative(question, memories), by_id))
        rows.append(_row(size, question, indexed_iterative(question, index), by_id))
        rows.append(_or_row(size, question, memories, by_id))
        rows.append(_hierarchical_row(size, question, hierarchical_index, by_id))
    return rows


def summarize(rows: list[ScaleRow]) -> list[dict]:
    groups: dict[tuple[int, str], list[ScaleRow]] = {}
    for row in rows:
        groups.setdefault((row.corpus_size, row.strategy), []).append(row)

    output = []
    for (size, strategy), group in sorted(groups.items()):
        output.append({
            "corpus_size": size,
            "strategy": strategy,
            "questions": len(group),
            "mean_recall": round(sum(r.recall for r in group) / len(group), 3),
            "full_evidence_rate": round(sum(r.all_evidence_retrieved for r in group) / len(group), 3),
            "mean_context_tokens": round(sum(r.context_tokens for r in group) / len(group), 1),
            "mean_candidates_scanned": round(sum(r.candidates_scanned for r in group) / len(group), 1),
            "mean_retrieval_rounds": round(sum(r.retrieval_rounds for r in group) / len(group), 1),
            "mean_latency_ms": round(sum(r.latency_ms for r in group) / len(group), 3),
        })
    return output


def run_scaling_experiment(sizes, questions, corpus_builder):
    rows: list[ScaleRow] = []
    build_times = []
    index_build_times = []
    for size in sizes:
        start = perf_counter()
        memories = corpus_builder(size)
        build_times.append((size, (perf_counter() - start) * 1000))
        start = perf_counter()
        InvertedIndex(memories)
        index_build_times.append((size, (perf_counter() - start) * 1000))
        rows.extend(run_scale_point(size, questions, memories))
    return rows, build_times, index_build_times


def row_dicts(rows):
    return [asdict(row) for row in rows]
