from __future__ import annotations

import re
from dataclasses import dataclass

from .data import BenchmarkQuestion, MemoryChunk


@dataclass(frozen=True)
class SufficiencyDecision:
    sufficient: bool
    confidence: float
    reason: str


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2}


def heuristic_sufficiency(
    question: BenchmarkQuestion,
    retrieved: list[MemoryChunk],
) -> SufficiencyDecision:
    """Estimate whether the current context is likely sufficient without gold evidence.

    This is intentionally conservative and deterministic. It is a baseline for the
    later LLM-judge experiment, not a production confidence model.
    """
    if not retrieved:
        return SufficiencyDecision(False, 0.0, "no_evidence")

    query_tokens = _tokens(question.query)
    evidence_tokens = set().union(*(_tokens(m.text) for m in retrieved))
    lexical_coverage = len(query_tokens & evidence_tokens) / max(1, len(query_tokens))

    # Multiple supporting chunks are useful for complex/update/temporal questions.
    support = len(retrieved)
    complex_bonus = 0.08 if question.complexity == "complex" and support >= 3 else 0.0
    confidence = min(1.0, lexical_coverage + complex_bonus)

    if lexical_coverage >= 0.65 and (question.complexity == "simple" or support >= 3):
        return SufficiencyDecision(True, confidence, "strong_lexical_coverage")

    return SufficiencyDecision(False, confidence, "insufficient_lexical_coverage")


class LLMJudge:
    """Optional LLM evidence-sufficiency judge.

    Kept behind a tiny interface so the retrieval engine does not depend on a
    particular provider. The MVP does not call a model automatically.
    """

    def __init__(self, client, model: str):
        self.client = client
        self.model = model

    def judge(self, question: BenchmarkQuestion, retrieved: list[MemoryChunk]) -> SufficiencyDecision:
        evidence = "\n".join(f"[{m.id}] {m.text}" for m in retrieved)
        prompt = (
            "Decide whether the evidence below is sufficient to answer the question. "
            "Return JSON with keys sufficient (boolean), confidence (0-1), reason (string). "
            "Do not assume facts that are not present in the evidence.\n\n"
            f"Question: {question.query}\n\nEvidence:\n{evidence}"
        )
        response = self.client.responses.create(model=self.model, input=prompt)
        import json

        data = json.loads(response.output_text)
        return SufficiencyDecision(
            bool(data["sufficient"]), float(data["confidence"]), str(data["reason"])
        )
