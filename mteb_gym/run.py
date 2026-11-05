"""
One call runs the pipeline:

    corpus -> queries -> mteb retrieval per model -> judging -> ranking -> record

The results folder has the layout of the results repository, so publishing a run is copying its files:

    <output_folder>/<task>/<task>__<judge>__<queries>__q<n>-s<seed>-<hash>.json    one record per run
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import NamedTuple

from . import corpus as corpus_mod
from . import results, retrieval
from . import task as task_mod
from .cache import (
    grades_key,
    judge_pair_cached,
    judge_pool_cached,
    model_id,
    resolve_cache_folder,
    sha,
    verdict_key,
)
from .judge import Grade, PairwiseJudge, PointwiseJudge, Verdict, is_decision_model
from .llm import llm_settings
from .queries import Query, QueryGenerator
from .rank import ModelRating, rate, rate_pointwise
from .results import Result
from .retrieval import Ranked, slug
from .task_instructions import TASK_INSTRUCTIONS

logger = logging.getLogger(__name__)

MAX_JUDGE_FAILURE_RATE = 0.5


def resolve_instruction(task_instruction: str | None, corpus) -> tuple[str, str]:
    """The sentence the generator and the judge are given, and where it came from."""
    if task_instruction:
        return task_instruction, "config:task_instruction"
    if corpus.name in TASK_INSTRUCTIONS:
        return TASK_INSTRUCTIONS[corpus.name], "mteb_gym:task_instructions"
    raise ValueError(
        f"{corpus.name}: pass task_instruction, one sentence on what a good result is, for example "
        "'Given a support ticket, retrieve the help article that resolves it'."
    )


def _read_query_set(path: Path) -> tuple[list[Query], int | None, dict | None]:
    data = json.loads(path.read_text())
    return [Query(**q) for q in data["queries"]], data["n_generated"], data["generator"]


def _cached_queries(
    path: Path, gen: QueryGenerator, docs: dict[str, str]
) -> tuple[list[Query], int | None, dict | None]:
    """The query set and the generator settings it was written with. When two processes generate the same set, the
    first to finish wins and the other reads its queries."""
    if path.exists():
        return _read_query_set(path)
    queries = gen.run(docs)
    if not queries:
        raise RuntimeError(
            "query generation kept 0 queries: the generator's answers were not parseable JSON, "
            "or the filter rejected all of them (see the log)"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"generator": gen.settings, "n_generated": gen.n_generated, "queries": [asdict(q) for q in queries]}
    temporary = path.with_suffix(f".tmp{os.getpid()}-{threading.get_ident()}")
    temporary.write_text(json.dumps(data, indent=2))
    try:
        os.link(temporary, path)  # fails if the file exists, and never leaves a partial one
    except FileExistsError:
        return _read_query_set(path)
    finally:
        temporary.unlink()
    return queries, gen.n_generated, gen.settings


def sample_queries(queries: dict[str, str], n: int, seed: int) -> dict[str, str]:
    """`n` of the queries, drawn with `seed`, in their order. All of them when there are no more than `n`."""
    if len(queries) <= n:
        return queries
    keep = set(random.Random(seed).sample(sorted(queries), n))
    return {qid: text for qid, text in queries.items() if qid in keep}


def own_queries(spec) -> list[Query]:
    """Your own queries: a list of strings, a .txt (one per line) or a .jsonl with id/text."""
    if isinstance(spec, (list, tuple)):
        return [Query(f"q{i}", str(t), []) for i, t in enumerate(spec)]
    path = Path(spec)
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in lines]
        return [Query(str(r.get("id", i)), str(r["text"]), []) for i, r in enumerate(rows)]
    return [Query(f"q{i}", line.strip(), []) for i, line in enumerate(lines)]


def pair_subset(n_pairs: int, qids: list[str], k: int | None, seed: int) -> dict[str, set[int]] | None:
    """For each query, the indices of the `k` model pairs to judge, drawn once with `seed`.
    Bradley-Terry ranks from incomplete comparisons, so this cuts judging cost by n_pairs / k."""
    if k is None or k >= n_pairs:
        return None
    rng = random.Random(seed)
    return {qid: set(rng.sample(range(n_pairs), k)) for qid in qids}


class QuerySet(NamedTuple):
    """The queries a run judges on, and how they were obtained."""

    id: str  # part of the prediction and verdict paths
    texts: dict[str, str]
    queries: list[Query] | None  # None when they are the dataset's own
    query_source: str  # synthetic | original | own
    n_generated: int | None  # before filtering
    generator_settings: dict | None


def _texts_hash(texts: dict[str, str]) -> str:
    return sha(*(f"{k}:{v}" for k, v in texts.items()))


def resolve_queries(corp, cache: Path, queries, gen: QueryGenerator, n_queries: int, seed: int) -> QuerySet:
    """Generate the queries, take the dataset's own, or read the ones you supplied."""
    if queries == "synthetic":
        recipe = f"{slug(corp.id)}-{slug(model_id(gen.client))}-{sha(sorted(gen.params.items()), *gen.prompts)}"
        qs, n_generated, settings = _cached_queries(cache / "queries" / corp.name / f"{recipe}.json", gen, corp.docs)
        texts = {q.qid: q.text for q in qs}
        return QuerySet(f"{recipe}-{_texts_hash(texts)}", texts, qs, "synthetic", n_generated, settings)
    if queries == "original":
        if not corp.queries:
            raise ValueError(f"{corp.name} has no queries of its own")
        positives = {
            q: t for q, t in corp.queries.items() if any(v > 0 for v in (corp.qrels or {}).get(q, {}).values())
        }
        texts = sample_queries(positives, n_queries, seed)  # mteb drops a query without a relevant document
        return QuerySet(f"{slug(corp.id)}-original-n{len(texts)}-s{seed}", texts, None, "original", None, None)
    qs = own_queries(queries)
    texts = {q.qid: q.text for q in qs}
    return QuerySet(f"{slug(corp.id)}-own-{_texts_hash(texts)}", texts, qs, "own", None, None)


def grade_diagnostics(grades: list[Grade]) -> dict:
    n = len(grades)
    failed = sum(not g.parsed_ok for g in grades)
    per_query: dict[str, int] = {}
    for g in grades:
        per_query[g.qid] = per_query.get(g.qid, 0) + 1
    ok = [g.grade for g in grades if g.parsed_ok]
    return {
        "judge_calls": n,
        "n_grades": n,
        "pool_size_mean": n / len(per_query) if per_query else None,
        "grade_mean": sum(ok) / len(ok) if ok else None,
        "parse_failure_rate": failed / n if n else None,
    }


def _check_judging(diagnostics: dict, ratings: list[ModelRating], models: list[str]) -> None:
    """Fail rather than write a record the judge did not really produce."""
    failure_rate = diagnostics.get("parse_failure_rate") or 0.0
    if failure_rate > MAX_JUDGE_FAILURE_RATE:
        raise RuntimeError(f"{failure_rate:.0%} of the judge's answers failed: check the endpoint and its answers")
    missing = sorted(set(models) - {r.name for r in ratings})
    if missing:
        raise RuntimeError(f"no usable judgement for {missing}: check the endpoint and its answers")


def _resolve_judging(judging: str, judge, generator, queries) -> str:
    """The judging mode, after checking the judge can do it."""
    if judging == "auto":
        judging = "pointwise" if is_decision_model(judge) else "pairwise"
    if judging not in ("pointwise", "pairwise"):
        raise ValueError(f"judging must be 'auto', 'pointwise' or 'pairwise', not {judging!r}")
    if is_decision_model(judge) and judging == "pairwise":
        raise ValueError("a decision model judges pointwise; pass an LLM to judge pairwise")
    if is_decision_model(judge) and generator is None and queries == "synthetic":
        raise ValueError("a decision model cannot write queries: pass generator=, or queries='original' / your own")
    return judging


def _judge_pointwise(
    judge,
    instruction: str,
    corp,
    ranked: dict[str, list[Ranked]],
    cache: Path,
    query_set: str,
    top_k,
    doc_chars,
    seed,
    workers,
) -> tuple[list[ModelRating], dict, str]:
    """Grade every pooled document once and rate each model by nDCG over the grades."""
    pj = PointwiseJudge(judge, instruction=instruction, workers=workers, doc_chars=doc_chars)
    key = grades_key(model_id(pj.client), pj.system, pj.instruction, pj.doc_chars, query_set)
    grades = judge_pool_cached(cache / "grades" / corp.name, pj, ranked, corp.docs, key, query_set)
    diagnostics = grade_diagnostics(grades)
    # a query with an ungraded document is left out: counting the document as 0 would punish whichever model found it
    failed = {g.qid for g in grades if not g.parsed_ok}
    diagnostics["queries_left_out"] = len(failed)
    if failed:
        logger.warning("left out %d queries with a document the judge could not grade", len(failed))
    usable = {(g.qid, g.doc_id): g.grade for g in grades if g.qid not in failed}
    kept = {m: [r for r in lists if r.qid not in failed] for m, lists in ranked.items()}
    if not usable:
        raise RuntimeError("the judge graded no query completely: check the endpoint and its answers")
    return rate_pointwise(usable, kept, k=top_k, seed=seed), diagnostics, pj.system


def _judge_pairwise(
    judge,
    instruction: str,
    corp,
    models: list[str],
    revisions: dict,
    ranked: dict[str, list[Ranked]],
    cache: Path,
    query_set: str,
    texts: dict[str, str],
    top_k,
    doc_chars,
    pairs_per_query,
    seed,
    workers,
) -> tuple[list[ModelRating], list[Verdict], str]:
    """Compare the models' lists pair by pair, and rate them with Bradley-Terry."""
    # One limit on LLM calls in flight: `workers` pairs at once, each judged one query at a time.
    # Pairs write separate verdict files, so they are independent.
    jd = PairwiseJudge(judge, instruction=instruction, workers=1, doc_chars=doc_chars)
    pairs = list(itertools.combinations(models, 2))
    chosen = pair_subset(len(pairs), list(texts), pairs_per_query, seed)  # None: every pair for every query

    def judge_pair(i: int) -> list[Verdict]:
        a, b = pairs[i]
        ra, rb = ranked[a], ranked[b]
        if chosen is not None:
            ra = [r for r in ra if i in chosen[r.qid]]
            rb = [r for r in rb if i in chosen[r.qid]]
        key = verdict_key(
            model_id(jd.client), jd.system, top_k, jd.doc_chars, query_set, a, revisions[a], b, revisions[b]
        )
        return judge_pair_cached(cache / "verdicts" / corp.name, jd, a, b, ra, rb, key, query_set)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        # map keeps pair order, so the verdict list, and the bootstrap over it, is the same every run
        verdicts = [v for pair in pool.map(judge_pair, range(len(pairs))) for v in pair]
    logger.info("judged %d pairs", len(pairs))
    return rate(verdicts, seed=seed), verdicts, jd.system


def predict(
    corpus: str | Path,
    model: str,
    *,
    generator=None,
    queries="synthetic",
    task_instruction: str | None = None,
    n_queries: int = 100,
    seed: int = 0,
    filter_queries: bool = True,
    cache_folder: str | Path | None = None,
    batch_size: int = 32,
    workers: int = 8,
) -> Path:
    """Retrieve one model's results for `corpus`, and write mteb's prediction file.

    One model per call, as `mteb run` takes one model, so a roster can be run one model per process
    and the GPU is returned to the operating system between them. `run()` then finds every
    prediction file already written and only judges. Arguments that decide the query set must match
    the later `run()`, since predictions are stored per query set.

    Returns:
        The path of mteb's prediction file.
    """
    import mteb

    if queries == "synthetic" and generator is None:
        raise ValueError(
            "predict() needs the generator that run() will use: a generated query set is identified by it, "
            "and predictions are stored per query set. Pass generator=, or queries='original' / your own."
        )
    cache = resolve_cache_folder(cache_folder)
    corp = corpus_mod.load(corpus)
    instruction, _ = resolve_instruction(task_instruction, corp)
    logger.info("criterion: %s", instruction)
    gen = QueryGenerator(
        generator,
        task_instruction=instruction,
        n_queries=n_queries,
        seed=seed,
        filter_queries=filter_queries,
        workers=workers,
    )
    qset = resolve_queries(corp, cache, queries, gen, n_queries, seed)
    logger.info("query set: %s", qset.id)
    revision = mteb.get_model_meta(model).revision
    folder = cache / "predictions" / corp.name / f"{slug(model)}@{revision}" / qset.id
    return retrieval.predict(model, task_mod.build(corp, qset.queries, qset.texts), folder, batch_size=batch_size)


def run(
    corpus: str | Path,
    models: list[str],
    judge,
    generator=None,
    *,
    queries="synthetic",
    task_instruction: str | None = None,
    n_queries: int = 100,
    top_k: int = 10,
    doc_chars: int = 2000,
    pairs_per_query: int | None = None,
    judging: str = "auto",
    seed: int = 0,
    filter_queries: bool = True,
    output_folder: str | Path = "results",
    cache_folder: str | Path | None = None,
    batch_size: int = 32,
    workers: int = 8,
) -> Result:
    """Rank `models` on `corpus` without relevance labels.

    Args:
        corpus: An mteb retrieval task name, a folder of .txt/.md files, or a .jsonl with id and text.
        models: mteb model ids. They run through mteb itself.
        judge: A decision model (mteb_gym.DecisionModel, e.g. JEV-27B), which grades each retrieved document, or an
            LLM client (mteb_gym.LLM or MockLLM), which compares the retrieved lists pairwise.
        generator: An LLM client that writes the queries. Defaults to the judge.
        queries: "synthetic" (generated), "original" (the dataset's own queries), or your own as a
            .jsonl with id and text, a .txt with one query per line, or a list of strings.
        task_instruction: One sentence on what a good result is, given to generator and judge. Required, except for
            the MTEB tasks in mteb_gym.task_instructions.
        n_queries: Queries to generate.
        top_k: Documents judged per query.
        doc_chars: Characters of each document shown to the judge; longer documents are cut and marked.
        pairs_per_query: Pairwise judging: judge only this many random model pairs per query; all pairs by default.
        judging: "pointwise" grades every pooled document once (0-5) and ranks models by nDCG@top_k over the grades;
            "pairwise" compares the models' lists and ranks with Bradley-Terry. "auto": pointwise for a decision
            model, pairwise for an LLM.
        seed: Seed for document sampling and the bootstrap.
        filter_queries: LLM quality filter and deduplication of generated queries.
        output_folder: Where the record is written, as <task>/<record>.json: the results repository's layout.
        cache_folder: Where queries, predictions, grades and verdicts are cached. Defaults to $MTEB_GYM_CACHE,
            else ~/.cache/mteb_gym.
        batch_size: Encoding batch size.
        workers: Concurrent grades when judging pointwise, or model pairs when judging pairwise.

    Returns:
        The run's Result: leaderboard, record and path.
    """
    started = time.time()
    out, models = Path(output_folder), sorted(set(models))  # the order must not change what is judged or its identity
    cache = resolve_cache_folder(cache_folder)
    if not models:
        raise ValueError("no models given")
    judging = _resolve_judging(judging, judge, generator, queries)
    gen_client = generator if generator is not None else judge
    if generator is None and queries == "synthetic":
        logger.warning("no generator given: the judge also writes the queries (self-preference risk)")

    corp = corpus_mod.load(corpus)
    instruction, instruction_source = resolve_instruction(task_instruction, corp)
    logger.info("criterion: %s", instruction)
    gen = QueryGenerator(
        gen_client,
        task_instruction=instruction,
        n_queries=n_queries,
        seed=seed,
        filter_queries=filter_queries,
        workers=workers,
    )

    qset = resolve_queries(corp, cache, queries, gen, n_queries, seed)
    logger.info("query set: %s", qset.id)
    query_set, texts, qs, query_source = qset.id, qset.texts, qset.queries, qset.query_source

    import mteb

    revisions = {m: mteb.get_model_meta(m).revision for m in models}  # mteb's pins: part of every cache identity
    gym_task = task_mod.build(corp, qs, texts)
    labels = task_mod.labels(corp, qs, texts)  # None for your own queries
    ranked: dict[str, list[Ranked]] = {}
    ndcg: dict[str, float] = {}  # the baseline that needs no judge: nDCG@10 against the labels
    for m in models:
        folder = cache / "predictions" / corp.name / f"{slug(m)}@{revisions[m]}" / query_set
        pred = retrieval.predict(m, gym_task, folder, batch_size=batch_size)
        ranked[m] = retrieval.top_k(pred, corp, texts, top_k)
        if labels:
            ndcg[m] = retrieval.ndcg_at_10(pred, labels, corp.ignore_identical_ids)

    verdicts: list[Verdict] = []
    if judging == "pointwise":
        ratings, diagnostics, judge_system = _judge_pointwise(
            judge, instruction, corp, ranked, cache, query_set, top_k, doc_chars, seed, workers
        )
    else:
        ratings, verdicts, judge_system = _judge_pairwise(
            judge,
            instruction,
            corp,
            models,
            revisions,
            ranked,
            cache,
            query_set,
            texts,
            top_k,
            doc_chars,
            pairs_per_query,
            seed,
            workers,
        )
        diagnostics = None

    _check_judging(diagnostics or results.verdict_diagnostics(verdicts), ratings, models)
    experiment = {
        "corpus_id": corp.id,
        "query_set": query_set,
        "query_source": query_source,
        "judge_model": model_id(judge),
        "generator_model": model_id(gen_client) if query_source == "synthetic" else None,
        "task_instruction": instruction,
        "task_instruction_source": instruction_source,
        "judge_system": judge_system,
        "n_queries_generated": qset.n_generated,
        "n_queries": len(texts),
        "top_k": top_k,
        "doc_chars": doc_chars,
        "pairs_per_query": pairs_per_query if judging == "pairwise" else None,
        "judging": judging,
        "seed": seed,
        **({f"gen_{k}": v for k, v in gen.params.items()} if query_source == "synthetic" else {}),
        "models": models,
        "model_revisions": revisions,
    }
    experiment["config_hash"] = results.config_hash(experiment)
    path = results.record_path(out, corp.name, experiment)
    llms = {"judge": llm_settings(judge), "generator": qset.generator_settings}
    label_source = "dataset" if query_source == "original" else "seed_documents" if labels else None
    result = Result(
        results.build_record(
            corp,
            experiment,
            ratings,
            verdicts,
            time.time() - started,
            revisions,
            llms,
            ndcg=ndcg,
            labels=label_source,
            diagnostics=diagnostics,
        ),
        path,
    )
    if path.exists():
        existing = Result.from_disk(path)
        if existing.record["ratings"] == result.record["ratings"]:  # it may hold an agreement added since
            return existing
    result.to_disk()
    return result
