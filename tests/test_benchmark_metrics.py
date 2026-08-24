from context_memory.benchmark_metrics import retrieval_metrics


def test_hit_and_recall_at_k_are_reported_separately():
    metrics = retrieval_metrics(("m1", "noise", "m2", "m3"), ("m1", "m3"))

    assert metrics.hit_at_1 == 1.0
    assert metrics.hit_at_3 == 1.0
    assert metrics.recall_at_1 == 0.5
    assert metrics.recall_at_3 == 0.5
    assert metrics.recall_at_5 == 1.0


def test_no_hit_is_zero():
    metrics = retrieval_metrics(("noise", "noise2"), ("m1",))

    assert metrics.hit_at_1 == 0.0
    assert metrics.hit_at_3 == 0.0
    assert metrics.recall_at_8 == 0.0
