from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryChunk:
    id: str
    text: str


@dataclass(frozen=True)
class BenchmarkQuestion:
    id: str
    category: str
    query: str
    gold_evidence_ids: tuple[str, ...]
    complexity: str


MEMORIES = [
    MemoryChunk("m1", "The API uses PostgreSQL as the primary transactional database."),
    MemoryChunk("m2", "Redis was considered for caching but initially rejected because traffic was low."),
    MemoryChunk("m3", "In March, traffic increased enough to revisit the caching decision."),
    MemoryChunk("m4", "The team introduced Redis as the production cache in June."),
    MemoryChunk("m5", "Kafka was rejected for request/response traffic because synchronous HTTP was sufficient."),
    MemoryChunk("m6", "Kafka is used for asynchronous event ingestion after the June architecture update."),
    MemoryChunk("m7", "The architecture review said the new event pipeline reduced coupling between services."),
    MemoryChunk("m8", "The API timeout target is 200 ms at the p95 percentile."),
]

QUESTIONS = [
    BenchmarkQuestion(
        "q1", "single-hop", "What is the primary transactional database?", ("m1",), "simple"
    ),
    BenchmarkQuestion(
        "q2", "update", "Are we using Redis in production now?", ("m2", "m4"), "complex"
    ),
    BenchmarkQuestion(
        "q3", "temporal", "When did the team introduce Redis as the production cache?", ("m3", "m4"), "complex"
    ),
    BenchmarkQuestion(
        "q4", "update", "Is Kafka used in the current architecture?", ("m5", "m6"), "complex"
    ),
    BenchmarkQuestion(
        "q5", "multi-hop", "Why was Kafka introduced for the event pipeline?", ("m6", "m7"), "complex"
    ),
]
