from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Protocol

from .data import BenchmarkQuestion, MemoryChunk


@dataclass(frozen=True)
class JudgeDecision:
    sufficient: bool
    confidence: float
    missing: tuple[str, ...] = ()


class SufficiencyJudge(Protocol):
    def judge(self, question: BenchmarkQuestion, memories: list[MemoryChunk]) -> JudgeDecision: ...


class OpenAISufficiencyJudge:
    """LLM evidence judge. Requires the optional `llm` dependency and OPENAI_API_KEY."""

    def __init__(self, model: str | None = None):
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model or os.environ.get("OPENAI_MODEL")
        if not self.model:
            raise ValueError("Set OPENAI_MODEL when using --llm-judge")

    def judge(self, question: BenchmarkQuestion, memories: list[MemoryChunk]) -> JudgeDecision:
        evidence = "\n".join(f"[{m.id}] {m.text}" for m in memories)
        prompt = f"""You are an evidence sufficiency judge for a long-term memory system.

Question:
{question.query}

Retrieved memories:
{evidence}

Decide whether the retrieved memories contain enough information to answer the question accurately.
Do not assume facts that are not present in the memories.
If information is missing, list concise search concepts that could help retrieve it.

Return ONLY JSON matching this schema:
{{"sufficient": true|false, "confidence": 0.0, "missing": ["concept"]}}
"""
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
        )
        payload = json.loads(response.choices[0].message.content or "{}")
        return JudgeDecision(
            bool(payload.get("sufficient", False)),
            float(payload.get("confidence", 0.0)),
            tuple(str(x) for x in payload.get("missing", [])),
        )
