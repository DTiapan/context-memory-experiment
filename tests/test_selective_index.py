from context_memory.data import MemoryChunk
from context_memory.selective_index import SelectiveMemoryIndex


def memory(memory_id: str, text: str) -> MemoryChunk:
    return MemoryChunk(
        memory_id,
        text,
        "test",
        "fact",
        (),
        "2026-01",
    )


def test_selective_index_intersects_postings():
    memories = [
        memory("m1", "Redis caching reduced API latency"),
        memory("m2", "Redis caching was evaluated"),
        memory("m3", "Postgres database migration improved reliability"),
    ]
    index = SelectiveMemoryIndex(memories)
    result = index.lookup("Redis caching")
    assert result.count == 2
    assert set(result.ids) == {"m1", "m2"}


def test_selectivity_is_candidate_ratio():
    memories = [memory(f"m{i}", "redis caching") for i in range(10)]
    memories += [memory("other", "kafka events")]
    index = SelectiveMemoryIndex(memories)
    assert index.selectivity("redis") == 10 / 11
