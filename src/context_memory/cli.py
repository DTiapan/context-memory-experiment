import argparse
import json

from .data import MEMORIES, QUESTIONS
from .evaluation_v4 import run_benchmark_v4
from .judge import OpenRouterSufficiencyJudge
from .evidence_directed import KeywordSufficiencyJudge


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-judge", action="store_true", help="run the OpenRouter evidence judge")
    parser.add_argument("--model", default=None, help="OpenRouter model; defaults to OPENROUTER_MODEL or Nemotron 3.5 Lightning Free")
    args = parser.parse_args()

    judge = OpenRouterSufficiencyJudge(args.model) if args.llm_judge else KeywordSufficiencyJudge()
    rows, summaries, category_summaries, stopping_summaries = run_benchmark_v4(QUESTIONS, MEMORIES, judge)
    print(json.dumps({
        "judge": "openrouter" if args.llm_judge else "keyword",
        "model": args.model,
        "summary": summaries,
        "by_category": category_summaries,
        "by_stopping_reason": stopping_summaries,
        "rows": [r.__dict__ for r in rows],
    }, indent=2))


if __name__ == "__main__":
    main()
