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
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

from . import corpus as corpus_mod
from . import results, retrieval
from . import task as task_mod
from .cache import (
    capped,
    grades_key,
    judge_pair_cached,
    judge_pool_cached,
    model_id,
    prediction_folder,
    resolve_cache_folder,
    sha,
    texts_hash,
    verdict_file,
    verdict_key,
    verdicts_done,
)
from .judge import PAIRWISE_DOC_CHARS, PairwiseJudge, PointwiseJudge, Verdict, is_decision_model
from .llm import ChatClient, llm_settings, load_model
from .local import load_now
from .models.models_protocols import DecisionClient
from .progress import progress_bar
from .queries import DOC_CHARS, Query, QueryGenerator
from .rank import ModelRating, rate, rate_pointwise
from .results import Result
from .retrieval import Ranked, slug

logger = logging.getLogger(__name__)

MAX_JUDGE_FAILURE_RATE = 0.5
MAX_QUERIES_LEFT_OUT = 0.1  # of a run's queries, for lack of a complete set of grades
MIN_MODEL_COVERAGE = 0.5  # share of a model's comparisons that must have a usable answer


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
    except OSError:  # no hard links on this filesystem
        temporary.replace(path)
        return queries, gen.n_generated, gen.settings
    finally:
        temporary.unlink(missing_ok=True)
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
        queries = [Query(f"q{i}", str(t), []) for i, t in enumerate(spec)]
    else:
        path = Path(spec)
        lines = [line for line in path.read_text().split("\n") if line.strip()]
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in lines]
            queries = [Query(str(r.get("id", i)), str(r["text"]), []) for i, r in enumerate(rows)]
        else:
            queries = [Query(f"q{i}", line.strip(), []) for i, line in enumerate(lines)]
    if not queries:
        raise ValueError("no queries given")
    return queries


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


def resolve_queries(corp, cache: Path, queries, gen: QueryGenerator, n_queries: int, seed: int) -> QuerySet:
    """Generate the queries, take the dataset's own, or read the ones you supplied."""
    if queries == "synthetic":
        recipe = f"{slug(corp.id)}-{slug(model_id(gen.client))}-{sha(sorted(gen.params.items()), *gen.prompts)}"
        qs, n_generated, settings = _cached_queries(cache / "queries" / corp.name / f"{recipe}.json", gen, corp.docs)
        texts = {q.qid: q.text for q in qs}
        return QuerySet(f"{recipe}-{texts_hash(texts)}", texts, qs, "synthetic", n_generated, settings)
    if queries == "original":
        if not corp.queries:
            raise ValueError(f"{corp.name} has no queries of its own")
        positives = {
            q: t for q, t in corp.queries.items() if any(v > 0 for v in (corp.qrels or {}).get(q, {}).values())
        }
        if not positives:
            raise ValueError(f"{corp.name} has no query with a relevant document")
        texts = sample_queries(positives, n_queries, seed)  # mteb drops a query without a relevant document
        limited = "-top-ranked" if corp.top_ranked else ""  # what these queries may retrieve is part of their identity
        return QuerySet(f"{slug(corp.id)}-original-n{len(texts)}-s{seed}{limited}", texts, None, "original", None, None)
    qs = own_queries(queries)
    texts = {q.qid: q.text for q in qs}
    return QuerySet(f"{slug(corp.id)}-own-{texts_hash(texts)}", texts, qs, "own", None, None)


def _check_judging(diagnostics: dict, ratings: list[ModelRating], models: list[str]) -> None:
    """Raise if the judging is too incomplete to rank on."""
    failure_rate = diagnostics.get("parse_failure_rate") or 0.0
    if failure_rate > MAX_JUDGE_FAILURE_RATE:
        raise RuntimeError(f"{failure_rate:.0%} of the judge's answers failed: check the endpoint and its answers")
    if (diagnostics.get("identical_retrieval_rate") or 0.0) > 0.95:
        raise RuntimeError("the models retrieved identical lists: the corpus is likely empty or smaller than top_k")
    missing = sorted(set(models) - {r.name for r in ratings})
    if missing:
        raise RuntimeError(f"no usable judgement for {missing}: check the endpoint and its answers")


def _check_coverage(verdicts: list[Verdict]) -> None:
    """Raise if failed answers left a model with too few comparisons to rate fairly."""
    asked: Counter = Counter()
    usable: Counter = Counter()
    for v in verdicts:
        for m in (v.model_a, v.model_b):
            asked[m] += 1
            usable[m] += all(v.parsed_ok)
    thin = sorted(m for m in asked if usable[m] / asked[m] < MIN_MODEL_COVERAGE)
    if thin:
        raise RuntimeError(f"most comparisons failed for {thin}: check the endpoint and its answers")


def _load(role: str, model, url: str | None):
    """A judge or generator by name, saying what it turned out to be: on stdout, so a Python caller sees it too."""
    client = load_model(model, url)
    if client is not None:
        name = model if isinstance(model, str) else model_id(client)
        print(f"{role}: {name} ({getattr(client, 'how', 'your own object')})")
    return client


def _resolve_judging(judging: str, judge, generator, queries) -> str:
    """The judging mode, after checking the judge can do it."""
    if judging not in ("pointwise", "pairwise"):
        raise ValueError(f"judging must be 'pointwise' or 'pairwise', not {judging!r}")
    if is_decision_model(generator):
        raise ValueError("a decision model cannot write queries: choose a chat model as the generator")
    if is_decision_model(judge) and generator is None and queries == "synthetic":
        raise ValueError("a decision model cannot write queries: pass generator=, or queries='original' / your own")
    return judging


@dataclass(frozen=True)
class _Judging:
    """What both kinds of judging need to know about the run."""

    corpus: object
    cache: Path
    query_set: str
    top_k: int
    doc_chars: int
    seed: int
    workers: int
    show_progress_bar: bool = False


def _judge_pointwise(
    judge, instruction: str | None, ranked: dict[str, list[Ranked]], run: _Judging
) -> tuple[list[ModelRating], dict, str]:
    """Grade every pooled document once and rate each model by nDCG over the grades."""
    pj = PointwiseJudge(judge, instruction=instruction, workers=run.workers, doc_chars=run.doc_chars)
    key = grades_key(model_id(pj.client), pj.system, pj.instruction, pj.doc_chars, run.query_set)
    folder = run.cache / "grades" / run.corpus.name
    with progress_bar("Judging", 0, run.show_progress_bar, unit="grade") as bar:
        grades = judge_pool_cached(folder, pj, ranked, run.corpus.docs, key, run.query_set, bar)
    diagnostics = results.grade_diagnostics(grades)
    # A query with an ungraded document is left out: counting it as 0 would punish whichever model found it.
    failed = {g.qid for g in grades if not g.parsed_ok}
    n_queries = len({g.qid for g in grades})
    diagnostics.update(queries_left_out=len(failed), queries_rated=n_queries - len(failed))
    if n_queries and len(failed) / n_queries > MAX_QUERIES_LEFT_OUT:
        raise RuntimeError(
            f"the judge could not grade {len(failed)} of {n_queries} queries completely: "
            "check the endpoint and its answers"
        )
    if failed:
        logger.warning("left out %d queries with a document the judge could not grade", len(failed))
    usable = {(g.qid, g.doc_id): g.grade for g in grades if g.qid not in failed}
    kept = {m: [r for r in lists if r.qid not in failed] for m, lists in ranked.items()}
    return rate_pointwise(usable, kept, k=run.top_k, seed=run.seed), diagnostics, pj.system


def _judge_pairwise(
    judge,
    instruction: str | None,
    ranked: dict[str, list[Ranked]],
    revisions: dict[str, str | None],
    pairs_per_query: int | None,
    run: _Judging,
) -> tuple[list[ModelRating], dict, str]:
    """Compare the models' lists pair by pair, and rate them with Bradley-Terry."""
    # One limit on LLM calls in flight: `workers` pairs at once, each judged one query at a time.
    jd = PairwiseJudge(judge, instruction=instruction, workers=1, doc_chars=run.doc_chars)
    pairs = list(itertools.combinations(sorted(ranked), 2))
    qids = [r.qid for r in next(iter(ranked.values()))]
    chosen = pair_subset(len(pairs), qids, pairs_per_query, run.seed)  # None: every pair for every query

    folder = run.cache / "verdicts" / run.corpus.name
    jobs = []
    for i, (a, b) in enumerate(pairs):
        ra, rb = ranked[a], ranked[b]
        if chosen is not None:
            ra = [r for r in ra if i in chosen[r.qid]]
            rb = [r for r in rb if i in chosen[r.qid]]
        key = verdict_key(
            model_id(jd.client), jd.system, run.top_k, jd.doc_chars, run.query_set, a, revisions[a], b, revisions[b]
        )
        jobs.append((a, b, ra, rb, key))
    if any(set(r.qid for r in ra) - set(verdicts_done(verdict_file(folder, a, b, key))) for a, b, ra, _, key in jobs):
        load_now(judge)  # before the bar, so it times only the judging

    def judge_pair(job) -> list[Verdict]:
        return judge_pair_cached(folder, jd, *job, run.query_set, bar)

    with (
        progress_bar("Judging", sum(len(job[2]) for job in jobs), run.show_progress_bar, unit="comparison") as bar,
        ThreadPoolExecutor(max_workers=run.workers) as pool,
    ):
        # map keeps pair order, so the verdict list, and the bootstrap over it, is the same every run
        verdicts = [v for pair in pool.map(judge_pair, jobs) for v in pair]
    logger.info("judged %d pairs", len(pairs))
    _check_coverage(verdicts)
    return rate(verdicts, seed=run.seed), results.verdict_diagnostics(verdicts), jd.system


def predict(
    corpus: str | Path,
    model: str,
    *,
    generator: str | ChatClient | None = None,
    generator_url: str | None = None,
    queries: str | Path | list[str] = "synthetic",
    task_instruction: str | None = None,
    n_queries: int = 100,
    doc_chars: int | None = None,
    seed: int = 0,
    filter_queries: bool = False,
    cache_folder: str | Path | None = None,
    embedding_cache: str | Path | None = None,
    max_seq_length: int | None = None,
    batch_size: int = 32,
    workers: int = 8,
    show_progress_bar: bool = True,
) -> Path:
    """Retrieve one model's results for `corpus`, and write mteb's prediction file.

    One model per call, as `mteb run` takes one model, so a roster can be run one model per process
    and the GPU is returned to the operating system between them. `run()` then finds every
    prediction file already written and only judges. Arguments that decide the query set must match
    the later `run()`, since predictions are stored per query set, and `max_seq_length`, since they are stored per cap.

    Returns:
        The path of mteb's prediction file.
    """
    import mteb

    if queries == "synthetic" and generator is None:
        raise ValueError(
            "predict() needs the generator that run() will use: a generated query set is identified by it, "
            "and predictions are stored per query set. Pass generator=, or queries='original' / your own."
        )
    generator = _load("generator", generator, generator_url)
    if is_decision_model(generator):
        raise ValueError("a decision model cannot write queries: choose a chat model as the generator")
    cache = resolve_cache_folder(cache_folder)
    corp = corpus_mod.load(corpus)
    gen = QueryGenerator(
        generator,
        task_instruction=task_instruction or None,
        n_queries=n_queries,
        seed=seed,
        filter_queries=filter_queries,
        workers=workers,
        doc_chars=doc_chars or DOC_CHARS,
        show_progress_bar=show_progress_bar,
    )
    qset = resolve_queries(corp, cache, queries, gen, n_queries, seed)
    logger.info("query set: %s", qset.id)
    revision = mteb.get_model_meta(model).revision
    folder = prediction_folder(cache, corp.name, model, revision, qset.id, max_seq_length)
    return retrieval.predict(
        model,
        task_mod.build(corp, qset.queries, qset.texts),
        folder,
        batch_size=batch_size,
        embedding_cache=embedding_cache,
        max_seq_length=max_seq_length,
    )


def run(
    corpus: str | Path,
    models: list[str],
    judge: str | ChatClient | DecisionClient,
    generator: str | ChatClient | None = None,
    *,
    judge_url: str | None = None,
    generator_url: str | None = None,
    queries: str | Path | list[str] = "synthetic",
    task_instruction: str | None = None,
    n_queries: int = 100,
    top_k: int = 10,
    doc_chars: int | None = None,
    pairs_per_query: int | None = None,
    judging: str = "pointwise",
    seed: int = 0,
    filter_queries: bool = False,
    output_folder: str | Path = "results",
    cache_folder: str | Path | None = None,
    embedding_cache: str | Path | None = None,
    max_seq_length: int | None = None,
    batch_size: int = 32,
    workers: int = 8,
    show_progress_bar: bool = True,
) -> Result:
    """Rank `models` on `corpus` without relevance labels.

    Args:
        corpus: An mteb retrieval task name, a folder of .txt/.md files, or a .jsonl with id and text.
        models: mteb model ids. They run through mteb itself.
        judge: A model name: a chat model or a decision model, loaded directly or served.
        generator: The name of a chat model that writes the queries. Defaults to the judge.
        judge_url: Where the judge is served, if not an API. An OpenAI-compatible endpoint or a decision model's.
        generator_url: Where the generator is served, if not an API.
        queries: "synthetic" (generated), "original" (the dataset's own queries), or your own as a
            .jsonl with id and text, a .txt with one query per line, or a list of strings.
        task_instruction: Optional sentence describing the retrieval task, given to the generator and the judge. Without it
            the generator writes natural queries and the judge grades plain relevance. Recommended.
            mteb_gym.TASK_INSTRUCTIONS has one for each of a set of MTEB tasks.
        n_queries: Queries to generate.
        top_k: Documents judged per query.
        doc_chars: Characters of each document shown to the generator and the judge. Defaults to 16000, with a pairwise
            judge on 4000, since a comparison shows 2 * top_k documents, which costs far more per call.
        pairs_per_query: Pairwise judging: judge only this many random model pairs per query; all pairs by default.
        judging: "pointwise" grades every pooled document once (0-5) and ranks models by nDCG@top_k over the grades;
            "pairwise" compares the models' lists and ranks with Bradley-Terry.
        seed: Seed for document sampling and the bootstrap.
        filter_queries: Add an LLM quality filter and deduplication of generated queries.
        output_folder: Where the record is written, as <task>/<record>.json: the results repository's layout.
        cache_folder: Where queries, predictions, grades and verdicts are cached. Defaults to $MTEB_GYM_CACHE,
            else ~/.cache/mteb_gym.
        embedding_cache: A folder for mteb's embedding cache, so each corpus is encoded once per model. It only reuses
            vectors, so it changes no query set, prediction folder or record.
        max_seq_length: Cap the length of a SentenceTransformer that allows more; a static model and any other model are
            unaffected. By default models run as mteb loads them. A cap changes what a model retrieves, so it is part of
            the predictions' and verdicts' identity, and recorded in the record's config.
        batch_size: Encoding batch size.
        workers: Concurrent grades when judging pointwise, or model pairs when judging pairwise.
        show_progress_bar: Show a bar for writing queries, retrieving and judging. Work already cached counts as done.

    Returns:
        The run's Result: leaderboard, record and path.
    """
    started = time.time()
    out, models = Path(output_folder), sorted(set(models))  # the order must not change what is judged or its identity
    cache = resolve_cache_folder(cache_folder)
    if not models:
        raise ValueError("no models given")
    judge, generator = _load("judge", judge, judge_url), _load("generator", generator, generator_url)
    judging = _resolve_judging(judging, judge, generator, queries)
    gen_client = generator if generator is not None else judge
    if generator is None and queries == "synthetic":
        logger.warning("no generator given: the judge also writes the queries")

    corp = corpus_mod.load(corpus)
    instruction = task_instruction or None
    if instruction is None:
        logger.info(
            "no task_instruction: natural queries and plain relevance; a sentence describing the retrieval task helps"
        )
    gen_doc_chars = doc_chars or DOC_CHARS
    doc_chars = doc_chars or (PAIRWISE_DOC_CHARS if judging == "pairwise" else DOC_CHARS)
    gen = QueryGenerator(
        gen_client,
        task_instruction=instruction,
        n_queries=n_queries,
        seed=seed,
        filter_queries=filter_queries,
        workers=workers,
        doc_chars=gen_doc_chars,
        show_progress_bar=show_progress_bar,
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
    with progress_bar("Retrieving", len(models), show_progress_bar, unit="model") as bar:
        for m in models:
            bar.set_postfix_str(m)
            folder = prediction_folder(cache, corp.name, m, revisions[m], query_set, max_seq_length)
            pred = retrieval.predict(
                m,
                gym_task,
                folder,
                batch_size=batch_size,
                embedding_cache=embedding_cache,
                max_seq_length=max_seq_length,
            )
            ranked[m] = retrieval.top_k(pred, corp, texts, top_k)
            if labels:
                ndcg[m] = retrieval.ndcg_at_10(pred, labels, corp.ignore_identical_ids)
            bar.update()

    judging_run = _Judging(corp, cache, query_set, top_k, doc_chars, seed, workers, show_progress_bar)
    if judging == "pointwise":
        ratings, diagnostics, judge_system = _judge_pointwise(judge, instruction, ranked, judging_run)
    else:
        ratings, diagnostics, judge_system = _judge_pairwise(
            judge,
            instruction,
            ranked,
            {m: capped(r, max_seq_length) for m, r in revisions.items()},
            pairs_per_query,
            judging_run,
        )
    _check_judging(diagnostics, ratings, models)
    experiment = {
        "corpus_id": corp.id,
        "query_set": query_set,
        "query_source": query_source,
        "judge_model": model_id(judge),
        "generator_model": model_id(gen_client) if query_source == "synthetic" else None,
        "task_instruction": instruction,
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
        **({"max_seq_length": max_seq_length} if max_seq_length is not None else {}),
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
            diagnostics,
            time.time() - started,
            revisions,
            llms,
            ndcg=ndcg,
            labels=label_source,
        ),
        path,
    )
    if path.exists():
        existing = Result.from_disk(path)
        if existing.record["ratings"] == result.record["ratings"]:  # it may hold an agreement added since
            return existing
    result.to_disk()
    return result
