from context_memory.data import MemoryChunk
from context_memory.hierarchical_index import HierarchicalMemoryIndex


def memory(mid, text, topic, entities):
    return MemoryChunk(mid, text, topic, "fact", tuple(entities), "2026-01")


def test_query_selects_topic_group_before_memory_lookup():
    memories = [
        memory("m1", "Redis cache decision", "caching", ["redis", "caching"]),
        memory("m2", "Redis cache rollout", "caching", ["redis", "caching"]),
        memory("m3", "Kafka event pipeline", "messaging", ["kafka", "events"]),
    ]
    result = HierarchicalMemoryIndex(memories).lookup("Redis caching")
    assert result.group_ids == ("caching",)
    assert result.memory_ids == ("m1", "m2")


def test_unmatched_query_falls_back_to_all_groups():
    memories = [
        memory("m1", "Redis cache", "caching", ["redis"]),
        memory("m2", "Kafka events", "messaging", ["kafka"]),
    ]
    result = HierarchicalMemoryIndex(memories).lookup("quantum computing")
    assert set(result.group_ids) == {"caching", "messaging"}
    assert result.count == 2


def test_group_limit_bounds_candidate_groups():
    memories = [
        memory("m1", "Redis cache", "caching", ["redis"]),
        memory("m2", "Kafka events", "messaging", ["kafka"]),
        memory("m3", "Postgres database", "database", ["postgresql"]),
    ]
    result = HierarchicalMemoryIndex(memories).lookup("redis kafka postgres", max_groups=2)
    assert len(result.group_ids) == 2
