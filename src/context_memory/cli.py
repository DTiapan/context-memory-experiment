import json

from .data import MEMORIES, QUESTIONS
from .evaluation import run_benchmark


def main() -> None:
    rows, summaries, category_summaries = run_benchmark(QUESTIONS, MEMORIES)
    print(
        json.dumps(
            {
                "summary": summaries,
                "by_category": category_summaries,
                "rows": [r.__dict__ for r in rows],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
