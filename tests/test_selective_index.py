from context_memory.data import MemoryChunk
from context_memory.selective_index import SelectiveMemoryIndex


def test_selective_index_intersects_postings():
    memories = [
        MemoryChunk("m1", "Redis caching reduced API latency"),
        MemoryChunk("m2", "Redis caching was evaluated"),
        MemoryChunk("m3", "Postgres database migration improved reliability"),
    ]
    index = SelectiveMemoryIndex(memories)
    result = index.lookup("Redis caching")
    assert result.count == 2
    assert set(result.ids) == {"m1", "m2"}


def test_selectivity_is_candidate_ratio():
    memories = [MemoryChunk(f"m{i}", "redis caching") for i in range(10)]
    memories += [MemoryChunk("other", "kafka events")]
    index = SelectiveMemoryIndex(memories)
    assert index.selectivity("redis") == 10 / 11
