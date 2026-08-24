from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalMetrics:
    """Retrieval metrics that map cleanly to common memory/RAG reporting."""

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


def _metrics_at_k(retrieved_ids: tuple[str, ...], gold_ids: tuple[str, ...], k: int) -> tuple[float, float, float]:
    gold = set(gold_ids)
    retrieved = tuple(retrieved_ids[:k])
    if not gold:
        # Empty gold means the question is intentionally unanswerable. Precision/recall
        # are undefined; the separate unsupported-query signal captures whether retrieval
        # nevertheless returned evidence.
        return 0.0, 0.0, 0.0
    retrieved_set = set(retrieved)
    hit = 1.0 if retrieved_set & gold else 0.0
    recall = len(retrieved_set & gold) / len(gold)
    precision = len(retrieved_set & gold) / len(retrieved) if retrieved else 0.0
    return hit, recall, precision


def retrieval_metrics(retrieved_ids: tuple[str, ...], gold_ids: tuple[str, ...]) -> RetrievalMetrics:
    values = [_metrics_at_k(retrieved_ids, gold_ids, k) for k in (1, 3, 5, 8)]
    unsupported = 1.0 if not gold_ids and bool(retrieved_ids) else 0.0
    return RetrievalMetrics(
        hit_at_1=values[0][0],
        hit_at_3=values[1][0],
        hit_at_5=values[2][0],
        hit_at_8=values[3][0],
        recall_at_1=values[0][1],
        recall_at_3=values[1][1],
        recall_at_5=values[2][1],
        recall_at_8=values[3][1],
        precision_at_1=values[0][2],
        precision_at_3=values[1][2],
        precision_at_5=values[2][2],
        precision_at_8=values[3][2],
        unsupported_query_retrieval=unsupported,
    )
