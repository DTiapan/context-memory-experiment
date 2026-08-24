# context-memory-experiment

Small experiments for testing whether indexed and adaptive memory retrieval can reduce LLM context while preserving the evidence needed to answer questions.

## MVP 1

This iteration compares four retrieval strategies:

1. **Full context** — pass every memory to the evaluator.
2. **Fixed Top-K** — score every memory and keep the top 2.
3. **Adaptive Top-K** — score every memory and use K=2 for simple queries or K=4 for complex queries.
4. **Indexed adaptive** — use an inverted index to narrow the candidate set first, then perform the same deeper ranking step.

The fourth strategy is the first direct test of the database-style **index → deeper lookup** idea.

## Run

```bash
python -m pip install -e .
python -m context_memory.cli
pytest -q
```

The benchmark reports:

- evidence recall
- full-evidence retrieval rate
- context tokens
- retrieved chunks
- candidates scanned
- retrieval latency

The token estimate is intentionally simple for now. A real tokenizer will be introduced when we add an LLM/vector retrieval baseline.

## Next experiments

- inspect failures across 30 controlled questions
- make adaptive retrieval evidence-driven rather than hard-coded K
- integrate a real semantic/vector index
- run a larger external benchmark such as LongMemEval
