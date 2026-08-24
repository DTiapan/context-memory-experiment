from context_memory.data import MemoryChunk
from context_memory.index_diagnostics import diagnose


def memory(memory_id: str, text: str) -> MemoryChunk:
    return MemoryChunk(memory_id, text, "test", "fact", (), "2026-01")


def test_diagnostics_report_posting_size_and_gold_presence():
    memories = [
        memory("m1", "Redis caching reduced latency"),
        memory("m2", "Redis caching was evaluated"),
        memory("m3", "Postgres database migration"),
    ]
    result = diagnose("Redis caching", memories, ("m1",))
    by_term = {item.term: item for item in result.terms}

    assert by_term["redis"].posting_count == 2
    assert by_term["redis"].contains_gold is True
    assert by_term["caching"].posting_count == 2
    assert result.candidate_count == 2
    assert result.gold_in_candidates is True
