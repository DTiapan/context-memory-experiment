from context_memory.data import MemoryChunk
from context_memory.or_retrieval import retrieve_or


def memory(memory_id: str, text: str) -> MemoryChunk:
    return MemoryChunk(memory_id, text, "test", "fact", (), "2026-01")


def test_or_retrieval_preserves_candidates_from_any_query_term():
    memories = [
        memory("m1", "distributed caching was introduced"),
        memory("m2", "caching was used from the beginning"),
        memory("m3", "unrelated database migration"),
    ]
    result = retrieve_or(
        "distributed caching beginning",
        memories,
        gold_evidence_ids=("m1", "m2"),
    )

    assert result.candidate_count == 2
    assert result.gold_evidence_in_candidates is True
    assert {item.memory.id for item in result.candidates} == {"m1", "m2"}


def test_or_retrieval_ranks_rare_term_matches_higher():
    memories = [
        memory("m1", "redis caching latency"),
        memory("m2", "redis caching"),
        memory("m3", "redis common"),
    ]
    result = retrieve_or("redis latency", memories, top_k=2)
    assert result.candidate_count == 3
    assert result.candidates[0].memory.id == "m1"
    assert "latency" in result.candidates[0].matched_terms
