from context_memory.data import MEMORIES, QUESTIONS
from context_memory.evaluation import evaluate_one
from context_memory.retrieval import InvertedIndex, adaptive, fixed_top_k, indexed_adaptive


def test_adaptive_uses_more_context_for_complex_queries():
    simple = QUESTIONS[0]
    complex_question = QUESTIONS[1]
    simple_result = adaptive(simple, MEMORIES)
    complex_result = adaptive(complex_question, MEMORIES)
    assert len(simple_result.chunk_ids) == 2
    assert len(complex_result.chunk_ids) == 4


def test_evaluation_recall_is_computed_from_gold_evidence():
    question = QUESTIONS[1]
    result = fixed_top_k(question, MEMORIES, k=2)
    memory_by_id = {m.id: m for m in MEMORIES}
    row = evaluate_one(question, result, memory_by_id)
    assert 0.0 <= row.recall <= 1.0
    assert row.retrieved_count == 2


def test_index_reduces_candidate_search_space():
    index = InvertedIndex(MEMORIES)
    result = indexed_adaptive(QUESTIONS[19], index)
    assert result.candidate_count < len(MEMORIES)
    assert result.candidate_count > 0


def test_indexed_retrieval_returns_expected_evidence_for_simple_query():
    index = InvertedIndex(MEMORIES)
    result = indexed_adaptive(QUESTIONS[0], index)
    assert "m1" in result.chunk_ids
