from context_memory.data import QUESTIONS
from context_memory.expanded_benchmark import EXPANDED_QUESTIONS, build_expanded_questions


def test_expanded_benchmark_has_ten_variants_per_base_question():
    assert len(QUESTIONS) == 36
    assert len(EXPANDED_QUESTIONS) == 360
    assert len({q.id for q in EXPANDED_QUESTIONS}) == 360


def test_expanded_benchmark_includes_abstention_cases():
    abstention = [q for q in QUESTIONS if q.category == "abstention"]
    assert len(abstention) == 6
    assert all(q.gold_evidence_ids == () for q in abstention)


def test_expanded_questions_preserve_gold_evidence_and_categories():
    expanded = build_expanded_questions()
    for base in QUESTIONS:
        variants = [q for q in expanded if q.id.startswith(f"{base.id}-")]
        assert len(variants) == 10
        assert {q.category for q in variants} == {base.category}
        assert {q.gold_evidence_ids for q in variants} == {base.gold_evidence_ids}


def test_expanded_questions_change_query_surface():
    expanded = build_expanded_questions()
    for base in QUESTIONS:
        variants = [q for q in expanded if q.id.startswith(f"{base.id}-")]
        assert all(q.query != base.query for q in variants)
        assert len({q.query for q in variants}) == 10
