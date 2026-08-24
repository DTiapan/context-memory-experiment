from context_memory.data import BenchmarkQuestion, MemoryChunk
from context_memory.selective_retrieval import retrieve


def memory(memory_id: str, text: str) -> MemoryChunk:
    return MemoryChunk(memory_id, text, "test", "fact", (), "2026-01")


def question(query: str) -> BenchmarkQuestion:
    return BenchmarkQuestion("q", "single-hop", query, ("m1",), "simple")


def test_selective_retrieval_intersects_query_terms():
    memories = [
        memory("m1", "Redis caching reduced latency"),
        memory("m2", "Postgres database migration"),
        memory("m3", "Redis unrelated discussion"),
        memory("m4", "Caching unrelated discussion"),
    ]
    result = retrieve(question("Redis caching"), memories, top_k=2)
    assert result.candidate_count == 1
    assert result.candidate_ratio == 1 / 4
    assert result.memories[0].id == "m1"
