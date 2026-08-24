from context_memory.data import MEMORIES, QUESTIONS
from context_memory.retrieval import heuristic_iterative, indexed_heuristic_iterative, InvertedIndex


def test_heuristic_iterative_does_not_use_gold_evidence_to_stop():
    question = next(q for q in QUESTIONS if q.id == "q24")
    result = heuristic_iterative(question, MEMORIES)
    assert result.stopping_reason in {"heuristic_sufficient", "max_k_reached"}
    assert result.retrieval_rounds >= 1


def test_indexed_heuristic_iterative_is_available():
    question = next(q for q in QUESTIONS if q.id == "q24")
    result = indexed_heuristic_iterative(question, InvertedIndex(MEMORIES))
    assert result.strategy == "indexed_heuristic_iterative"
    assert result.candidate_count > 0


def test_heuristic_can_stop_before_full_corpus_for_simple_query():
    question = next(q for q in QUESTIONS if q.id == "q1")
    result = heuristic_iterative(question, MEMORIES)
    assert len(result.chunk_ids) < len(MEMORIES)
