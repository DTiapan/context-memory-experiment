from __future__ import annotations

import concurrent.futures
from dataclasses import asdict
from time import perf_counter

from .data import MEMORIES, QUESTIONS
from .evaluation import summarize
from .fixed import fixed_retrieval
from .hierarchical_index import HierarchicalIndex
from .indexed_iterative import indexed_iterative
from .iterative import iterative
from .or_ranked import or_ranked

DEFAULT_CORPUS_SIZES = (18, 100, 1_000, 10_000, 100_000)
DEFAULT_WORKERS = 3


def _scale_one(corpus_size: int) -> list[dict]:
    memories = [MEMORIES[i % len(MEMORIES)] for i in range(corpus_size)]
    index = HierarchicalIndex(memories)
    rows: list[dict] = []
    strategies = (
        ("fixed", lambda q: fixed_retrieval(q, memories)),
        ("hierarchical", lambda q: index.retrieve(q)),
        ("indexed_iterative", lambda q: indexed_iterative(q, index)),
        ("iterative", lambda q: iterative(q, memories)),
        ("or_ranked", lambda q: or_ranked(q, memories)),
    )
    for strategy, runner in strategies:
        for question in QUESTIONS:
            start = perf_counter()
            result = runner(question)
            latency_ms = (perf_counter() - start) * 1000
            row = asdict(summarize(question, result))
            row.update({"corpus_size": corpus_size, "strategy": strategy, "latency_ms": latency_ms})
            rows.append(row)
    return rows


def run_scale(corpus_sizes: tuple[int, ...] = DEFAULT_CORPUS_SIZES, max_workers: int = DEFAULT_WORKERS) -> list[dict]:
    """Run corpus scaling experiments in parallel by corpus size.

    max_workers is configurable so the benchmark can be tuned empirically on the host.
    """
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")
    sizes = tuple(corpus_sizes)
    workers = min(max_workers, len(sizes))
    print(f"[scale] running {len(sizes)} corpus sizes with {workers} workers", flush=True)
    completed = 0
    results_by_size: dict[int, list[dict]] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_scale_one, size): size for size in sizes}
        for future in concurrent.futures.as_completed(futures):
            size = futures[future]
            results_by_size[size] = future.result()
            completed += 1
            print(f"[scale] completed corpus={size:,} ({completed}/{len(sizes)})", flush=True)
    return [row for size in sizes for row in results_by_size[size]]
