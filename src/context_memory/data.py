from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryChunk:
    id: str
    text: str
    topic: str
    memory_type: str
    entities: tuple[str, ...]
    timestamp: str


@dataclass(frozen=True)
class BenchmarkQuestion:
    id: str
    category: str
    query: str
    gold_evidence_ids: tuple[str, ...]
    complexity: str


MEMORIES = [
    MemoryChunk("m1", "The API uses PostgreSQL as the primary transactional database.", "database", "decision", ("postgresql", "database"), "2026-01"),
    MemoryChunk("m2", "Redis was considered for caching but initially rejected because traffic was low.", "caching", "decision", ("redis", "caching"), "2026-01"),
    MemoryChunk("m3", "In March, traffic increased enough to revisit the caching decision.", "caching", "event", ("redis", "caching"), "2026-03"),
    MemoryChunk("m4", "The team introduced Redis as the production cache in June.", "caching", "decision", ("redis", "caching"), "2026-06"),
    MemoryChunk("m5", "Kafka was rejected for request/response traffic because synchronous HTTP was sufficient.", "messaging", "decision", ("kafka", "http"), "2026-02"),
    MemoryChunk("m6", "Kafka is used for asynchronous event ingestion after the June architecture update.", "messaging", "decision", ("kafka", "events"), "2026-06"),
    MemoryChunk("m7", "The architecture review said the new event pipeline reduced coupling between services.", "messaging", "rationale", ("kafka", "events", "architecture"), "2026-06"),
    MemoryChunk("m8", "The API timeout target is 200 ms at the p95 percentile.", "performance", "constraint", ("api", "latency"), "2026-02"),
    MemoryChunk("m9", "The team chose Python for the data-processing worker because the existing ML tooling was Python-first.", "runtime", "decision", ("python", "ml"), "2026-02"),
    MemoryChunk("m10", "A March load test showed the worker was CPU-bound during peak traffic.", "performance", "event", ("python", "worker"), "2026-03"),
    MemoryChunk("m11", "The worker was moved to a larger instance in April before any rewrite was attempted.", "runtime", "decision", ("python", "worker"), "2026-04"),
    MemoryChunk("m12", "The team kept PostgreSQL for transactional state after testing showed the query patterns remained relational.", "database", "rationale", ("postgresql", "database"), "2026-05"),
    MemoryChunk("m13", "The API gateway uses synchronous HTTP for client-facing requests.", "api", "fact", ("api", "http"), "2026-01"),
    MemoryChunk("m14", "Event consumers can retry failed messages without blocking client requests.", "messaging", "rationale", ("kafka", "events"), "2026-06"),
    MemoryChunk("m15", "The initial cache design used no distributed cache because the service had low request volume.", "caching", "decision", ("caching", "redis"), "2026-01"),
    MemoryChunk("m16", "After traffic growth, cache hit rate became a target for the API team.", "caching", "goal", ("redis", "caching", "api"), "2026-04"),
    MemoryChunk("m17", "The API p95 latency target was retained during the June architecture update.", "performance", "constraint", ("api", "latency"), "2026-06"),
    MemoryChunk("m18", "The architecture team documented event ingestion separately from request/response handling.", "messaging", "fact", ("kafka", "http", "events"), "2026-06"),
]


QUESTIONS = [
    BenchmarkQuestion("q1", "single-hop", "What is the primary transactional database?", ("m1",), "simple"),
    BenchmarkQuestion("q2", "single-hop", "What is the API timeout target?", ("m8",), "simple"),
    BenchmarkQuestion("q3", "single-hop", "What language was chosen for the data-processing worker?", ("m9",), "simple"),
    BenchmarkQuestion("q4", "single-hop", "What protocol is used for client-facing requests?", ("m13",), "simple"),
    BenchmarkQuestion("q5", "single-hop", "What became a target for the API team after traffic growth?", ("m16",), "simple"),
    BenchmarkQuestion("q6", "single-hop", "What database did the team retain after query-pattern testing?", ("m12",), "simple"),

    BenchmarkQuestion("q7", "update", "Are we using Redis in production now?", ("m2", "m4"), "complex"),
    BenchmarkQuestion("q8", "update", "Is Kafka used in the current architecture?", ("m5", "m6"), "complex"),
    BenchmarkQuestion("q9", "update", "Did the team rewrite the Python worker after it became CPU-bound?", ("m10", "m11"), "complex"),
    BenchmarkQuestion("q10", "update", "Did the API latency target change in June?", ("m8", "m17"), "complex"),
    BenchmarkQuestion("q11", "update", "Was distributed caching used from the beginning?", ("m15", "m4"), "complex"),
    BenchmarkQuestion("q12", "update", "Is PostgreSQL still used for transactional state?", ("m1", "m12"), "complex"),

    BenchmarkQuestion("q13", "temporal", "When was Redis introduced as the production cache?", ("m3", "m4"), "complex"),
    BenchmarkQuestion("q14", "temporal", "When did traffic growth cause the caching decision to be revisited?", ("m3", "m2"), "complex"),
    BenchmarkQuestion("q15", "temporal", "When did the worker become CPU-bound and what happened next?", ("m10", "m11"), "complex"),
    BenchmarkQuestion("q16", "temporal", "When was Kafka added for asynchronous event ingestion?", ("m6", "m18"), "complex"),
    BenchmarkQuestion("q17", "temporal", "When was the API latency target reaffirmed?", ("m8", "m17"), "complex"),
    BenchmarkQuestion("q18", "temporal", "When did the team decide to retain PostgreSQL?", ("m12",), "simple"),

    BenchmarkQuestion("q19", "multi-hop", "Why was Kafka introduced for the event pipeline?", ("m6", "m7"), "complex"),
    BenchmarkQuestion("q20", "multi-hop", "Why did the team introduce Redis after initially rejecting it?", ("m2", "m3", "m4"), "complex"),
    BenchmarkQuestion("q21", "multi-hop", "Why did the team avoid rewriting the worker when it became CPU-bound?", ("m9", "m10", "m11"), "complex"),
    BenchmarkQuestion("q22", "multi-hop", "Why is Kafka separated from client-facing HTTP traffic?", ("m5", "m6", "m18"), "complex"),
    BenchmarkQuestion("q23", "multi-hop", "Why was PostgreSQL retained for transactional state?", ("m1", "m12"), "complex"),
    BenchmarkQuestion("q24", "multi-hop", "How did traffic growth affect the caching architecture?", ("m2", "m3", "m4", "m16"), "complex"),

    BenchmarkQuestion("q25", "distractor-heavy", "What is the current cache technology?", ("m4",), "complex"),
    BenchmarkQuestion("q26", "distractor-heavy", "What is the current event-ingestion technology?", ("m6",), "complex"),
    BenchmarkQuestion("q27", "distractor-heavy", "What is the current transactional database decision?", ("m12",), "complex"),
    BenchmarkQuestion("q28", "distractor-heavy", "What is the current API latency constraint?", ("m17",), "complex"),
    BenchmarkQuestion("q29", "distractor-heavy", "What is the current worker deployment decision?", ("m11",), "complex"),
    BenchmarkQuestion("q30", "distractor-heavy", "What is the rationale for the current event pipeline?", ("m7", "m14"), "complex"),
]
