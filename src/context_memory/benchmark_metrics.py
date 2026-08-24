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


def _metrics_at_k(retrieved_ids: tuple[str, ...], gold_ids: tuple[str, ...], k: int) -> tuple[float, float]:
    gold = set(gold_ids)
    if not gold:
        return 0.0, 0.0
    retrieved = set(retrieved_ids[:k])
    hit = 1.0 if retrieved & gold else 0.0
    recall = len(retrieved & gold) / len(gold)
    return hit, recall


def retrieval_metrics(retrieved_ids: tuple[str, ...], gold_ids: tuple[str, ...]) -> RetrievalMetrics:
    values = [_metrics_at_k(retrieved_ids, gold_ids, k) for k in (1, 3, 5, 8)]
    return RetrievalMetrics(
        hit_at_1=values[0][0],
        hit_at_3=values[1][0],
        hit_at_5=values[2][0],
        hit_at_8=values[3][0],
        recall_at_1=values[0][1],
        recall_at_3=values[1][1],
        recall_at_5=values[2][1],
        recall_at_8=values[3][1],
    )
