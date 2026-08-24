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


class OpenRouterSufficiencyJudge:
    """LLM evidence judge using OpenRouter's OpenAI-compatible API."""

    def __init__(self, model: str | None = None):
        from openai import OpenAI

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("Set OPENROUTER_API_KEY when using --llm-judge")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "X-Title": "context-memory-experiment",
            },
        )
        self.model = model or os.environ.get("OPENROUTER_MODEL", "openai/gpt-5.5")

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


# Backward-compatible name for callers that still import the old class.
OpenAISufficiencyJudge = OpenRouterSufficiencyJudge
