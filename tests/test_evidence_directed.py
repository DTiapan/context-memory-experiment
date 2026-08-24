from context_memory.data import MEMORIES, QUESTIONS
from context_memory.evidence_directed import KeywordSufficiencyJudge, indexed_llm_iterative, llm_iterative
from context_memory.judge import JudgeDecision
from context_memory.retrieval import InvertedIndex


class AlwaysSufficientJudge:
    def judge(self, question, memories):
        return JudgeDecision(True, 1.0, ())


class AlwaysNeedsMoreJudge:
    def judge(self, question, memories):
        return JudgeDecision(False, 0.0, ("migration", "rationale"))


def test_evidence_directed_stops_when_judge_is_sufficient():
    question = QUESTIONS[0]
    result = llm_iterative(question, MEMORIES, AlwaysSufficientJudge())
    assert result.retrieval_rounds == 1
    assert result.stopping_reason == "judge_sufficient"
    assert result.chunk_ids


def test_evidence_directed_uses_missing_terms_for_follow_up_search():
    question = next(q for q in QUESTIONS if q.category == "multi-hop")
    result = llm_iterative(question, MEMORIES, AlwaysNeedsMoreJudge(), max_rounds=2)
    assert result.retrieval_rounds == 2
    assert result.stopping_reason == "max_rounds_reached"


def test_indexed_evidence_directed_can_stop_early():
    question = QUESTIONS[0]
    index = InvertedIndex(MEMORIES)
    result = indexed_llm_iterative(question, index, AlwaysSufficientJudge())
    assert result.retrieval_rounds == 1
    assert result.candidate_count < len(MEMORIES)


def test_keyword_judge_returns_a_bounded_confidence():
    judge = KeywordSufficiencyJudge()
    decision = judge.judge(QUESTIONS[0], [MEMORIES[0]])
    assert 0.0 <= decision.confidence <= 1.0
