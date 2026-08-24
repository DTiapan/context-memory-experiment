from __future__ import annotations

import cProfile
import io
import pstats
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from time import perf_counter

from .benchmark_metrics import retrieval_metrics
from .data import BenchmarkQuestion, MemoryChunk
from .retrieval import InvertedIndex, fixed_top_k, indexed_iterative, iterative
from .or_retrieval import retrieve_or
from .hierarchical_index import HierarchicalMemoryIndex
from .selective_index import SelectiveMemoryIndex
from .token_cache import TokenCache


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
    precision_at_1: float
    precision_at_3: float
    precision_at_5: float
    precision_at_8: float
    unsupported_query_retrieval: float
    all_evidence_retrieved: bool
    retrieval_rounds: int
    latency_ms: float


def _tokens(memories: list[MemoryChunk]) -> int:
    return sum(max(1, len(m.text.split())) for m in memories)


def _metrics_row(size, question, strategy, retrieved_ids, candidate_count, chunks, rounds, latency_ms):
    metrics = retrieval_metrics(retrieved_ids, question.gold_evidence_ids)
    gold = set(question.gold_evidence_ids)
    retrieved = set(retrieved_ids)
    return ScaleRow(
        size, question.id, question.category, strategy, len(retrieved_ids), candidate_count,
        _tokens(chunks), metrics.recall_at_8, metrics.hit_at_1, metrics.hit_at_3,
        metrics.hit_at_5, metrics.hit_at_8, metrics.recall_at_1, metrics.recall_at_3,
        metrics.recall_at_5, metrics.recall_at_8, metrics.precision_at_1,
        metrics.precision_at_3, metrics.precision_at_5, metrics.precision_at_8,
        metrics.unsupported_query_retrieval, gold.issubset(retrieved), rounds, latency_ms,
    )


def _row(size, question, result, by_id):
    return _metrics_row(size, question, result.strategy, result.chunk_ids, result.candidate_count,
                        [by_id[i] for i in result.chunk_ids], result.retrieval_rounds, result.latency_ms)


def _or_row(size, question, memories, by_id, selective_index):
    start = perf_counter()
    result = retrieve_or(question.query, memories, top_k=8, gold_evidence_ids=question.gold_evidence_ids,
                         index=selective_index)
    latency_ms = (perf_counter() - start) * 1000
    retrieved_ids = tuple(candidate.memory.id for candidate in result.candidates)
    return _metrics_row(size, question, "or_ranked", retrieved_ids, result.candidate_count,
                        [by_id[i] for i in retrieved_ids], 1, latency_ms)


def _hierarchical_row(size, question, index, by_id):
    start = perf_counter()
    candidate_set = index.lookup(question.query, max_groups=4)
    candidates = [index.memories[memory_id] for memory_id in candidate_set.memory_ids]
    query_terms = set(question.query.lower().split())
    scored = sorted(candidates, key=lambda memory: len(query_terms & set(memory.text.lower().split())), reverse=True)[:8]
    latency_ms = (perf_counter() - start) * 1000
    retrieved_ids = tuple(memory.id for memory in scored)
    return _metrics_row(size, question, "hierarchical", retrieved_ids, candidate_set.count, scored, 2, latency_ms)


def _question_rows(size, question, memories, index, hierarchical_index, by_id, token_cache, selective_index):
    return [
        _row(size, question, fixed_top_k(question, memories, token_cache=token_cache), by_id),
        _row(size, question, iterative(question, memories, token_cache=token_cache), by_id),
        _row(size, question, indexed_iterative(question, index), by_id),
        _or_row(size, question, memories, by_id, selective_index),
        _hierarchical_row(size, question, hierarchical_index, by_id),
    ]


def run_scale_point(size, questions, memories, progress=None, max_question_workers=1):
    token_cache = TokenCache(memories)
    index = InvertedIndex(memories, token_cache=token_cache)
    selective_index = SelectiveMemoryIndex(memories, token_cache=token_cache)
    hierarchical_index = HierarchicalMemoryIndex(memories)
    by_id = {m.id: m for m in memories}
    worker_count = max(1, min(max_question_workers, len(questions)))
    if worker_count == 1:
        rows = []
        for question_number, question in enumerate(questions, start=1):
            rows.extend(_question_rows(size, question, memories, index, hierarchical_index, by_id, token_cache, selective_index))
            if progress is not None:
                progress(question_number, len(questions), len(rows))
        return rows
    completed = 0
    rows_by_question = {}
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(_question_rows, size, question, memories, index, hierarchical_index, by_id, token_cache, selective_index):
                   (number, question.id) for number, question in enumerate(questions, start=1)}
        for future in as_completed(futures):
            number, question_id = futures[future]
            rows_by_question[number] = future.result()
            completed += 1
            if progress is not None:
                progress(completed, len(questions), completed * 5)
    rows = []
    for number in range(1, len(questions) + 1):
        rows.extend(rows_by_question[number])
    return rows


def _run_scale_point_worker(size, questions, corpus_builder, max_question_workers=1):
    start = perf_counter()
    memories = corpus_builder(size)
    build_ms = (perf_counter() - start) * 1000
    start = perf_counter()
    token_cache = TokenCache(memories)
    InvertedIndex(memories, token_cache=token_cache)
    index_build_ms = (perf_counter() - start) * 1000
    last_reported = 0
    started_at = perf_counter()

    def report(question_number, question_total, row_count):
        nonlocal last_reported
        if question_number != question_total and question_number - last_reported < 10:
            return
        last_reported = question_number
        elapsed = perf_counter() - started_at
        rate = question_number / elapsed if elapsed else 0.0
        remaining = (question_total - question_number) / rate if rate else 0.0
        print(f"[scale] corpus={size:,} | questions {question_number}/{question_total} | rows {row_count:,} | {rate:.1f} q/s | ETA {remaining:.1f}s", file=sys.stderr, flush=True)

    rows = run_scale_point(size, questions, memories, progress=report, max_question_workers=max_question_workers)
    return size, rows, build_ms, index_build_ms


def profile_scale_point(size, questions, corpus_builder, top_n=20):
    if size < 1:
        raise ValueError("size must be >= 1")
    if not questions:
        raise ValueError("questions must not be empty")
    if top_n < 1:
        raise ValueError("top_n must be >= 1")
    memories = corpus_builder(size)
    profiler = cProfile.Profile()
    started_at = perf_counter()
    profiler.enable()
    rows = []
    token_cache = TokenCache(memories)
    index = InvertedIndex(memories, token_cache=token_cache)
    selective_index = SelectiveMemoryIndex(memories, token_cache=token_cache)
    hierarchical_index = HierarchicalMemoryIndex(memories)
    by_id = {m.id: m for m in memories}
    for question in questions:
        rows.extend(_question_rows(size, question, memories, index, hierarchical_index, by_id, token_cache, selective_index))
    profiler.disable()
    elapsed_ms = (perf_counter() - started_at) * 1000
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative")
    stats.print_stats(top_n)
    return {"corpus_size": size, "questions_profiled": len(questions), "strategies_per_question": 5,
            "rows_generated": len(rows), "elapsed_ms": round(elapsed_ms, 3),
            "mean_question_ms": round(elapsed_ms / len(questions), 3), "profile": stream.getvalue()}


def summarize(rows):
    groups = {}
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
            "precision_at_1": round(sum(r.precision_at_1 for r in group) / len(group), 3),
            "precision_at_3": round(sum(r.precision_at_3 for r in group) / len(group), 3),
            "precision_at_5": round(sum(r.precision_at_5 for r in group) / len(group), 3),
            "precision_at_8": round(sum(r.precision_at_8 for r in group) / len(group), 3),
            "unsupported_query_retrieval_rate": round(sum(r.unsupported_query_retrieval for r in group) / len(group), 3),
            "full_evidence_rate": round(sum(r.all_evidence_retrieved for r in group) / len(group), 3),
            "mean_context_tokens": round(sum(r.context_tokens for r in group) / len(group), 1),
            "mean_candidates_scanned": round(sum(r.candidates_scanned for r in group) / len(group), 1),
            "mean_retrieval_rounds": round(sum(r.retrieval_rounds for r in group) / len(group), 1),
            "mean_latency_ms": round(sum(r.latency_ms for r in group) / len(group), 3),
        })
    return output


def run_scaling_experiment(sizes, questions, corpus_builder, max_workers=3, show_progress=True):
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")
    requested_workers = max_workers
    corpus_workers = min(max_workers, len(sizes))
    question_workers = max(1, (max_workers + corpus_workers - 1) // corpus_workers)
    rows_by_size, build_times, index_build_times = {}, {}, {}
    if show_progress:
        print(f"[scale] requested_workers={requested_workers} | corpus_jobs={len(sizes)} | corpus_workers={corpus_workers} | question_workers_per_corpus={question_workers}", file=sys.stderr, flush=True)
        if requested_workers > len(sizes):
            print("[scale] question-level parallelism enabled to use the remaining worker budget without changing retrieval algorithms", file=sys.stderr, flush=True)
    with ProcessPoolExecutor(max_workers=corpus_workers) as executor:
        futures = {executor.submit(_run_scale_point_worker, size, questions, corpus_builder, question_workers): size for size in sizes}
        completed = 0
        for future in as_completed(futures):
            size, point_rows, build_ms, index_build_ms = future.result()
            rows_by_size[size], build_times[size], index_build_times[size] = point_rows, build_ms, index_build_ms
            completed += 1
            if show_progress:
                print(f"[scale] completed corpus={size:,} ({completed}/{len(sizes)})", file=sys.stderr, flush=True)
    if show_progress:
        print("[scale] complete", file=sys.stderr, flush=True)
    return ([row for size in sizes for row in rows_by_size[size]],
            [(size, build_times[size]) for size in sizes],
            [(size, index_build_times[size]) for size in sizes])


def row_dicts(rows):
    return [asdict(row) for row in rows]
