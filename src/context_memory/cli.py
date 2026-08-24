import json

from .data import MEMORIES, QUESTIONS
from .evaluation_v3 import run_benchmark_v3


def main() -> None:
    rows, summaries, category_summaries, stopping_summaries = run_benchmark_v3(QUESTIONS, MEMORIES)
    print(
        json.dumps(
            {
                "summary": summaries,
                "by_category": category_summaries,
                "by_stopping_reason": stopping_summaries,
                "rows": [r.__dict__ for r in rows],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
