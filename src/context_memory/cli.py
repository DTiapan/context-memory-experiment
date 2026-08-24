import argparse
import json

from .data import MEMORIES, QUESTIONS
from .expanded_benchmark import EXPANDED_QUESTIONS
from .evaluation_v4 import run_benchmark_v4
from .evaluation_scale import run_scaling_experiment, summarize, row_dicts, profile_scale_point
from .diagnostic_scale import diagnostic_dicts
from .judge import OpenRouterSufficiencyJudge
from .evidence_directed import KeywordSufficiencyJudge
from .scaling import SCALE_SIZES, build_corpus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-judge", action="store_true", help="run the OpenRouter evidence judge")
    parser.add_argument("--model", default=None, help="OpenRouter model")
    parser.add_argument("--scale", action="store_true", help="run the corpus scaling experiment")
    parser.add_argument("--scale-size", type=int, default=None, help="run --scale for one supported corpus size")
    parser.add_argument("--profile", action="store_true", help="profile a small serial sample of the scale benchmark")
    parser.add_argument("--profile-size", type=int, default=100_000, help="corpus size for --profile (default: 100000)")
    parser.add_argument("--profile-questions", type=int, default=10, help="questions for --profile (default: 10)")
    parser.add_argument("--profile-top", type=int, default=20, help="top cProfile functions to report (default: 20)")
    parser.add_argument("--workers", type=int, default=3, help="total parallel worker budget for --scale (default: 3)")
    args = parser.parse_args()

    if args.profile:
        if args.profile_size < 1:
            parser.error("--profile-size must be >= 1")
        if args.profile_questions < 1:
            parser.error("--profile-questions must be >= 1")
        if args.profile_top < 1:
            parser.error("--profile-top must be >= 1")
        questions = EXPANDED_QUESTIONS[:args.profile_questions]
        print(f"[profile] corpus={args.profile_size:,} | questions={len(questions)} | strategies=5", flush=True)
        result = profile_scale_point(
            args.profile_size,
            questions,
            build_corpus,
            top_n=args.profile_top,
        )
        print(json.dumps(result, indent=2))
        return

    if args.scale:
        if args.workers < 1:
            parser.error("--workers must be >= 1")
        if args.scale_size is not None and args.scale_size not in SCALE_SIZES:
            parser.error(f"--scale-size must be one of {SCALE_SIZES}")

        scale_sizes = [args.scale_size] if args.scale_size is not None else list(SCALE_SIZES)
        scale_questions = EXPANDED_QUESTIONS
        corpus_workers = min(args.workers, len(scale_sizes))
        question_workers = max(1, (args.workers + corpus_workers - 1) // corpus_workers)
        rows, build_times, index_build_times = run_scaling_experiment(
            scale_sizes, scale_questions, build_corpus, max_workers=args.workers
        )
        diagnostic_rows = {}
        for size in scale_sizes:
            memories = build_corpus(size)
            diagnostic_rows[str(size)] = diagnostic_dicts(scale_questions, memories)

        print(json.dumps({
            "experiment": "corpus_scaling",
            "questions": len(scale_questions),
            "base_questions": len(QUESTIONS),
            "question_variants_per_base": len(scale_questions) // len(QUESTIONS),
            "corpus_sizes": scale_sizes,
            "workers_requested": args.workers,
            "parallelism": {
                "corpus_workers": corpus_workers,
                "question_workers_per_corpus": question_workers,
                "worker_budget": corpus_workers * question_workers,
            },
            "summary": summarize(rows),
            "corpus_build_ms": [{"corpus_size": s, "latency_ms": round(t, 3)} for s, t in build_times],
            "index_build_ms": [{"corpus_size": s, "latency_ms": round(t, 3)} for s, t in index_build_times],
            "diagnostics": diagnostic_rows,
            "rows": row_dicts(rows),
        }, indent=2))
        return

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
