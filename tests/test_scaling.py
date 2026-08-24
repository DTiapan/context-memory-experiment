from context_memory.data import MEMORIES, QUESTIONS
from context_memory.retrieval import InvertedIndex, iterative, indexed_iterative
from context_memory.scaling import SCALE_SIZES, build_corpus


def test_scaled_corpus_preserves_gold_memories():
    corpus = build_corpus(1000)
    assert len(corpus) == 1000
    assert [m.id for m in corpus[:18]] == [m.id for m in MEMORIES]
    assert len({m.id for m in corpus}) == 1000


def test_indexed_iterative_preserves_recall_on_scaled_corpus():
    corpus = build_corpus(1000)
    index = InvertedIndex(corpus)
    by_id = {m.id for m in corpus}
    for question in QUESTIONS:
        sequential = iterative(question, corpus)
        indexed = indexed_iterative(question, index)
        gold = set(question.gold_evidence_ids)
        assert gold <= by_id
        assert gold <= set(sequential.chunk_ids)
        assert gold <= set(indexed.chunk_ids)


def test_supported_scale_sizes_are_monotonic():
    assert SCALE_SIZES == (18, 100, 1_000, 10_000, 100_000)
