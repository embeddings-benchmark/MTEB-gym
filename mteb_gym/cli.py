"""The `mteb-gym` command."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import cast

import mteb
from mteb.abstasks.retrieval import AbsTaskRetrieval

from . import load_model, predict, run, submit
from .llm import Client


def _key(variable: str | None) -> str | None:
    return os.environ[variable] if variable else None


def client(model: str, base_url: str | None, extra_body: str | None = None, key_variable: str | None = None) -> Client:
    return load_model(model, base_url, _key(key_variable), json.loads(extra_body) if extra_body else None)


def _documents(path: Path) -> list[str]:
    """The texts of a .jsonl's rows, or of a folder's .txt and .md files."""
    if path.is_file() and path.suffix == ".jsonl":
        rows = (json.loads(line) for line in path.read_text().split("\n") if line.strip())
        texts = [str(r.get("text") or "") for r in rows]
    else:
        texts = [p.read_text(errors="ignore") for p in sorted(path.rglob("*")) if p.suffix in (".txt", ".md")]
    if not texts:
        raise ValueError(f"no documents found in {path}: a corpus is a folder of .txt or .md files, or a .jsonl")
    return texts


def _corpus_and_queries(ap: argparse.ArgumentParser) -> None:
    """The arguments that decide the query set and how models run: `predict` and `run` must agree on them."""
    ap.add_argument(
        "--corpus", type=str, required=True, help="mteb retrieval task name, or a directory / .jsonl of documents"
    )
    ap.add_argument("--generator", type=str, default=None, help="query generator model id (default: the judge)")
    ap.add_argument("--generator-url", type=str, default=None)
    ap.add_argument(
        "--generator-key-env", type=str, default=None, help="environment variable holding the generator's API key"
    )
    ap.add_argument("--generator-extra-body", type=str, default=None, help="JSON passed to the generator's server")
    ap.add_argument(
        "--queries",
        type=str,
        default="synthetic",
        help="'synthetic', 'original' (the dataset's own queries), or a path to your queries",
    )
    ap.add_argument(
        "--task-instruction",
        type=str,
        default=None,
        help="optional sentence describing the retrieval task, given to the generator and the judge (recommended)",
    )
    ap.add_argument(
        "--doc-chars",
        type=int,
        default=None,
        help="characters of each document shown to the generator and the judge (default: 16000, or 4000 judging pairwise)",
    )
    ap.add_argument("--n-queries", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--filter", action="store_true", help="add an LLM quality filter and deduplication of generated queries"
    )
    ap.add_argument(
        "--cache-folder",
        type=str,
        default=None,
        help="queries, predictions and judgments (default: $MTEB_GYM_CACHE, else ~/.cache/mteb_gym)",
    )
    ap.add_argument(
        "--embedding-cache", type=str, default=None, help="folder for mteb's embedding cache, to encode a corpus once"
    )
    ap.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help="cap a SentenceTransformer's length (default: each model as mteb loads it)",
    )
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-progress", action="store_true", help="no progress bars")


def _add_run_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    run_ap = sub.add_parser("run", help="rank models: queries, retrieval, judging, ranking")
    _corpus_and_queries(run_ap)
    run_ap.add_argument("--models", type=str, nargs="+", required=True, help="mteb model ids, e.g. mteb/baseline-bm25s")
    run_ap.add_argument(
        "--judge",
        type=str,
        required=True,
        help="judge model id on an OpenAI-compatible endpoint ('mock' for a dry run)",
    )
    run_ap.add_argument("--judge-url", type=str, default=None, help="base url for the judge endpoint")
    run_ap.add_argument(
        "--judge-key-env", type=str, default=None, help="environment variable holding the judge's API key"
    )
    run_ap.add_argument(
        "--judging",
        type=str,
        default="pointwise",
        choices=["pointwise", "pairwise"],
        help="pointwise: grade every pooled document, rank by nDCG; pairwise: compare lists",
    )
    run_ap.add_argument(
        "--judge-extra-body",
        type=str,
        default=None,
        help='JSON passed to the judge\'s server, e.g. \'{"chat_template_kwargs": {"enable_thinking": false}}\'',
    )
    run_ap.add_argument(
        "--output-folder",
        type=str,
        default="results",
        help="records, as <task>/<record>.json: the results repository's layout",
    )
    run_ap.add_argument("--top-k", type=int, default=10)
    run_ap.add_argument(
        "--pairs-per-query", type=int, default=None, help="judge only this many random model pairs per query"
    )


def _add_submit_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    sub_ap = sub.add_parser(
        "submit",
        help="records to the results repository on GitHub; their queries, predictions and judgments to the dataset on Hugging Face",
    )
    sub_ap.add_argument("--results-folder", type=str, default="results")
    sub_ap.add_argument("--cache-folder", type=str, default=None)
    sub_ap.add_argument(
        "--create-pr", action="store_true", help="open both pull requests; without it, nothing is uploaded"
    )


def _add_predict_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    pred_ap = sub.add_parser("predict", help="retrieve one model's results, so a roster runs one model per process")
    _corpus_and_queries(pred_ap)
    pred_ap.add_argument("--model", type=str, required=True, help="one mteb model id")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="mteb-gym", description="Rank embedding models on a corpus, without relevance labels."
    )
    sub = ap.add_subparsers(dest="command", required=True)

    _add_run_parser(sub)
    _add_submit_parser(sub)
    _add_predict_parser(sub)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if args.command == "submit":
        out = submit(args.results_folder, cache_folder=args.cache_folder, create_pr=args.create_pr)
        for key, where in (("pr_url", "results repository"), ("dataset_pr_url", "dataset")):
            if key in out:
                print(f"{where}: {out[key]}")
        return
    shared = dict(
        generator=client(args.generator, args.generator_url, args.generator_extra_body, args.generator_key_env)
        if args.generator
        else None,
        queries=args.queries,
        task_instruction=args.task_instruction,
        n_queries=args.n_queries,
        seed=args.seed,
        filter_queries=args.filter,
        doc_chars=args.doc_chars,
        cache_folder=args.cache_folder,
        embedding_cache=args.embedding_cache,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        workers=args.workers,
        show_progress_bar=not args.no_progress,
    )

    path = Path(args.corpus)
    corpus = _documents(path) if path.exists() else cast(AbsTaskRetrieval, mteb.get_task(args.corpus))
    if args.command == "predict":
        print(f"predictions -> {predict(corpus, mteb.get_model_meta(args.model), **shared)}")
        return

    judge = client(args.judge, args.judge_url, args.judge_extra_body, args.judge_key_env)
    result = run(
        corpus,
        [mteb.get_model_meta(m) for m in args.models],
        judge,
        judging=args.judging,
        top_k=args.top_k,
        pairs_per_query=args.pairs_per_query,
        output_folder=args.output_folder,
        **shared,
    )
    print(result.leaderboard)
    bias = result.record["diagnostics"].get("a_first_rate")
    if bias is not None:
        print(f"\nposition bias (a_first_rate): {bias:.2f}  (0.50 = unbiased)")
    print(f"record -> {result.path}")
