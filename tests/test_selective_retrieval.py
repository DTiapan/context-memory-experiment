from context_memory.data import BenchmarkQuestion, MemoryChunk
from context_memory.selective_retrieval import retrieve


def memory(memory_id: str, text: str) -> MemoryChunk:
    return MemoryChunk(memory_id, text, "test", "fact", (), "2026-01")


def question(query: str) -> BenchmarkQuestion:
    return BenchmarkQuestion("q", query, "single-hop", ("m1",))


def test_selective_retrieval_ranks_only_candidates():
    memories = [
        memory("m1", "Redis caching reduced latency"),
        memory("m2", "Postgres database migration"),
        memory("m3", "Redis unrelated discussion"),
    ]
    result = retrieve(question("Redis caching"), memories, top_k=2)
    assert result.candidate_count == 2
    assert result.candidate_ratio == 2 / 3
    assert result.memories[0].id == "m1"
