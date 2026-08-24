from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from time import perf_counter

from .benchmark_metrics import retrieval_metrics
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
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    hit_at_8: float
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_8: float
    all_evidence_retrieved: bool
    retrieval_rounds: int
    latency_ms: float


def _tokens(memories: list[MemoryChunk]) -> int:
    return sum(max(1, len(m.text.split())) for m in memories)


def _metrics_row(
    size: int,
    question: BenchmarkQuestion,
    strategy: str,
    retrieved_ids: tuple[str, ...],
    candidate_count: int,
    chunks: list[MemoryChunk],
    rounds: int,
    latency_ms: float,
) -> ScaleRow:
    metrics = retrieval_metrics(retrieved_ids, question.gold_evidence_ids)
    gold = set(question.gold_evidence_ids)
    retrieved = set(retrieved_ids)
    return ScaleRow(
        size,
        question.id,
        question.category,
        strategy,
        len(retrieved_ids),
        candidate_count,
        _tokens(chunks),
        metrics.recall_at_8,
        metrics.hit_at_1,
        metrics.hit_at_3,
        metrics.hit_at_5,
        metrics.hit_at_8,
        metrics.recall_at_1,
        metrics.recall_at_3,
        metrics.recall_at_5,
        metrics.recall_at_8,
        gold.issubset(retrieved),
        rounds,
        latency_ms,
    )


def _row(size: int, question: BenchmarkQuestion, result, by_id) -> ScaleRow:
    chunks = [by_id[i] for i in result.chunk_ids]
    return _metrics_row(
        size,
        question,
        result.strategy,
        result.chunk_ids,
        result.candidate_count,
        chunks,
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
    retrieved_ids = tuple(candidate.memory.id for candidate in result.candidates)
    chunks = [by_id[i] for i in retrieved_ids]
    return _metrics_row(size, question, "or_ranked", retrieved_ids, result.candidate_count, chunks, 1, latency_ms)


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
    retrieved_ids = tuple(memory.id for memory in scored)
    return _metrics_row(size, question, "hierarchical", retrieved_ids, candidate_set.count, scored, 2, latency_ms)


def run_scale_point(
    size: int,
    questions: list[BenchmarkQuestion],
    memories: list[MemoryChunk],
    progress=None,
) -> list[ScaleRow]:
    index = InvertedIndex(memories)
    hierarchical_index = HierarchicalMemoryIndex(memories)
    by_id = {m.id: m for m in memories}
    rows: list[ScaleRow] = []
    for question_number, question in enumerate(questions, start=1):
        rows.append(_row(size, question, fixed_top_k(question, memories), by_id))
        rows.append(_row(size, question, iterative(question, memories), by_id))
        rows.append(_row(size, question, indexed_iterative(question, index), by_id))
        rows.append(_or_row(size, question, memories, by_id))
        rows.append(_hierarchical_row(size, question, hierarchical_index, by_id))
        if progress is not None:
            progress(question_number, len(questions), len(rows))
    return rows


def _run_scale_point_worker(size, questions, corpus_builder):
    """Run one corpus size in a separate process so scale points overlap."""
    start = perf_counter()
    memories = corpus_builder(size)
    build_ms = (perf_counter() - start) * 1000

    start = perf_counter()
    InvertedIndex(memories)
    index_build_ms = (perf_counter() - start) * 1000

    last_reported = 0
    started_at = perf_counter()

    def report(question_number: int, question_total: int, row_count: int) -> None:
        nonlocal last_reported
        if question_number != question_total and question_number - last_reported < 10:
            return
        last_reported = question_number
        elapsed = perf_counter() - started_at
        rate = question_number / elapsed if elapsed else 0.0
        remaining = (question_total - question_number) / rate if rate else 0.0
        print(
            f"[scale] corpus={size:,} | questions {question_number}/{question_total} | "
            f"rows {row_count:,} | {rate:.1f} q/s | ETA {remaining:.1f}s",
            file=sys.stderr,
            flush=True,
        )

    rows = run_scale_point(size, questions, memories, progress=report)
    return size, rows, build_ms, index_build_ms


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
            "hit_at_1": round(sum(r.hit_at_1 for r in group) / len(group), 3),
            "hit_at_3": round(sum(r.hit_at_3 for r in group) / len(group), 3),
            "hit_at_5": round(sum(r.hit_at_5 for r in group) / len(group), 3),
            "recall_at_1": round(sum(r.recall_at_1 for r in group) / len(group), 3),
            "recall_at_3": round(sum(r.recall_at_3 for r in group) / len(group), 3),
            "recall_at_5": round(sum(r.recall_at_5 for r in group) / len(group), 3),
            "recall_at_8": round(sum(r.recall_at_8 for r in group) / len(group), 3),
            "full_evidence_rate": round(sum(r.all_evidence_retrieved for r in group) / len(group), 3),
            "mean_context_tokens": round(sum(r.context_tokens for r in group) / len(group), 1),
            "mean_candidates_scanned": round(sum(r.candidates_scanned for r in group) / len(group), 1),
            "mean_retrieval_rounds": round(sum(r.retrieval_rounds for r in group) / len(group), 1),
            "mean_latency_ms": round(sum(r.latency_ms for r in group) / len(group), 3),
        })
    return output


def run_scaling_experiment(sizes, questions, corpus_builder, show_progress: bool = True):
    """Run independent corpus sizes concurrently and collect results deterministically."""
    rows_by_size = {}
    build_times = {}
    index_build_times = {}
    total_points = len(sizes)
    max_workers = min(3, total_points)

    if show_progress:
        print(
            f"[scale] running {total_points} corpus sizes with {max_workers} workers",
            file=sys.stderr,
            flush=True,
        )

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_scale_point_worker, size, questions, corpus_builder): size
            for size in sizes
        }
        completed = 0
        for future in as_completed(futures):
            size, point_rows, build_ms, index_build_ms = future.result()
            rows_by_size[size] = point_rows
            build_times[size] = build_ms
            index_build_times[size] = index_build_ms
            completed += 1
            if show_progress:
                print(
                    f"[scale] completed corpus={size:,} ({completed}/{total_points})",
                    file=sys.stderr,
                    flush=True,
                )

    rows = []
    for size in sizes:
        rows.extend(rows_by_size[size])

    if show_progress:
        print("[scale] complete", file=sys.stderr, flush=True)

    return (
        rows,
        [(size, build_times[size]) for size in sizes],
        [(size, index_build_times[size]) for size in sizes],
    )


def row_dicts(rows):
    return [asdict(row) for row in rows]
