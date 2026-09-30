"""mteb-gym run --corpus NFCorpus --models mteb/baseline-bm25s BAAI/bge-base-en-v1.5 --judge gpt-4.1
mteb-gym predict --corpus NFCorpus --model BAAI/bge-base-en-v1.5 --generator gpt-4.1-mini"""

from __future__ import annotations

import argparse
import json
import logging

from . import LLM, MockLLM, predict, run, submit


def client(model: str, base_url: str | None, extra_body: str | None = None):
    if model == "mock":
        return MockLLM()
    return LLM(model, base_url=base_url, extra_body=json.loads(extra_body) if extra_body else None)


def _corpus_and_queries(ap: argparse.ArgumentParser) -> None:
    """The arguments that decide the query set: `predict` and `run` must agree on them."""
    ap.add_argument("--corpus", required=True, help="mteb retrieval task name, or a directory / .jsonl of documents")
    ap.add_argument("--generator", default=None, help="query generator model id (default: the judge)")
    ap.add_argument("--generator-url", default=None)
    ap.add_argument("--generator-extra-body", default=None, help="JSON passed to the generator's server")
    ap.add_argument(
        "--queries",
        default="synthetic",
        help="'synthetic', 'original' (the dataset's own queries), or a path to your queries",
    )
    ap.add_argument(
        "--task-description",
        default=None,
        help="one sentence on what counts as a good result (default: the task's mteb prompt)",
    )
    ap.add_argument("--n-queries", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-filter", action="store_true")
    ap.add_argument(
        "--cache-folder",
        default=None,
        help="queries, predictions and verdicts (default: $MTEB_GYM_CACHE, else ~/.cache/mteb_gym)",
    )
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="mteb-gym", description="Rank embedding models on a corpus with an LLM judge.")
    sub = ap.add_subparsers(dest="command", required=True)

    run_ap = sub.add_parser("run", help="rank models: queries, retrieval, judging, ranking")
    _corpus_and_queries(run_ap)
    run_ap.add_argument("--models", nargs="+", required=True, help="mteb model ids, e.g. mteb/baseline-bm25s")
    run_ap.add_argument(
        "--judge", required=True, help="judge model id on an OpenAI-compatible endpoint ('mock' for a dry run)"
    )
    run_ap.add_argument("--judge-url", default=None, help="base url for the judge endpoint")
    run_ap.add_argument(
        "--judge-extra-body",
        default=None,
        help='JSON passed to the judge\'s server, e.g. \'{"chat_template_kwargs": {"enable_thinking": false}}\'',
    )
    run_ap.add_argument(
        "--output-folder", default="results", help="records, as <task>/<record>.json: the results repository's layout"
    )
    run_ap.add_argument("--top-k", type=int, default=10)
    run_ap.add_argument("--doc-chars", type=int, default=2000, help="characters of each document shown to the judge")
    run_ap.add_argument(
        "--pairs-per-query", type=int, default=None, help="judge only this many random model pairs per query"
    )

    sub_ap = sub.add_parser("submit", help="open pull requests with new records and their queries and verdicts")
    sub_ap.add_argument("--results-folder", default="results")
    sub_ap.add_argument("--cache-folder", default=None)
    sub_ap.add_argument("--create-pr", action="store_true", help="push and open the pull requests")

    pred_ap = sub.add_parser("predict", help="retrieve one model's results, so a roster runs one model per process")
    _corpus_and_queries(pred_ap)
    pred_ap.add_argument("--model", required=True, help="one mteb model id")

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if args.command == "submit":
        out = submit(args.results_folder, cache_folder=args.cache_folder, create_pr=args.create_pr)
        for key in ("pr_url", "dataset_pr_url"):
            if key in out:
                print(out[key])
        return
    shared = dict(
        generator=client(args.generator, args.generator_url, args.generator_extra_body) if args.generator else None,
        queries=args.queries,
        task_description=args.task_description,
        n_queries=args.n_queries,
        seed=args.seed,
        filter_queries=not args.no_filter,
        cache_folder=args.cache_folder,
        batch_size=args.batch_size,
        workers=args.workers,
    )

    if args.command == "predict":
        print(f"predictions -> {predict(args.corpus, args.model, **shared)}")
        return

    result = run(
        args.corpus,
        args.models,
        client(args.judge, args.judge_url, args.judge_extra_body),
        top_k=args.top_k,
        doc_chars=args.doc_chars,
        pairs_per_query=args.pairs_per_query,
        output_folder=args.output_folder,
        **shared,
    )
    print(result.leaderboard)
    bias = result.record["diagnostics"]["a_first_rate"]
    if bias is not None:
        print(f"\nposition bias (a_first_rate): {bias:.2f}  (0.50 = unbiased)")
    print(f"record -> {result.path}")


if __name__ == "__main__":
    main()
