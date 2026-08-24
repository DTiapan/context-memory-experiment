from __future__ import annotations

from .data import MEMORIES, MemoryChunk


SCALE_SIZES = (18, 100, 1_000, 10_000, 100_000)

# These distractors deliberately reuse vocabulary from the benchmark while
# referring to different decisions/events. This prevents the index from
# looking artificially good on purely unrelated noise.
TOPICS = (
    ("database", ("postgresql", "mysql", "database", "transactional")),
    ("caching", ("redis", "cache", "caching", "traffic")),
    ("messaging", ("kafka", "events", "http", "pipeline")),
    ("performance", ("api", "latency", "p95", "timeout")),
    ("runtime", ("python", "worker", "cpu", "deployment")),
)


def _distractor(i: int) -> MemoryChunk:
    topic, terms = TOPICS[i % len(TOPICS)]
    a, b, c, d = terms
    phase = i % 12
    return MemoryChunk(
        id=f"noise-{i:06d}",
        text=(
            f"Experiment {i} recorded a {topic} discussion involving {a}, {b}, "
            f"{c}, and {d}. The team reviewed the option during phase {phase}, "
            f"but this note does not change the benchmark architecture decision."
        ),
        topic=topic,
        memory_type="distractor",
        entities=(a, b),
        timestamp=f"2025-{(i % 12) + 1:02d}",
    )


def build_corpus(size: int) -> list[MemoryChunk]:
    """Build a deterministic corpus containing the 18 gold memories plus noise."""
    if size < len(MEMORIES):
        raise ValueError(f"Corpus size must be >= {len(MEMORIES)}")
    if size not in SCALE_SIZES:
        raise ValueError(f"Supported sizes: {SCALE_SIZES}")
    return [*MEMORIES, *(_distractor(i) for i in range(size - len(MEMORIES)))]
