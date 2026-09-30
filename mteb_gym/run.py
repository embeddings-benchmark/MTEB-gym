"""
One call runs the pipeline:

    corpus -> queries -> mteb retrieval per model -> pairwise judging -> Bradley-Terry -> record

Results and caches are kept apart, as mteb keeps results apart from its cache. The results folder
has the layout of the results repository, so publishing a run is copying its files:

    <output_folder>/<task>/<task>__<judge>__<queries>__q<n>-s<seed>-<hash>.json    one record per run

The cache holds every stage a run can reuse (default ~/.cache/mteb_gym, or $MTEB_GYM_CACHE), laid
out like the dataset its queries and verdicts are published to:

    <cache_folder>/queries/<task>/<query set>.json            generated queries and their generator
    <cache_folder>/predictions/<task>/<model>@<revision>/<query set>/   mteb's prediction file
    <cache_folder>/verdicts/<task>/<pair>-<key>.jsonl         one line per comparison
"""

from __future__ import annotations

import hashlib
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
from .judge import Judge, Verdict, task_prompt
from .llm import llm_settings
from .queries import Query, QueryGenerator
from .rank import rate
from .results import Result
from .retrieval import Ranked, slug

logger = logging.getLogger(__name__)


def _sha(*parts) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12]


def _model_id(client) -> str:
    """The judge or generator as its answers depend on it: the model, and the knobs it was built
    with. A temperature, an output cap, and extra_body server settings such as thinking mode."""
    name = str(getattr(client, "model", type(client).__name__))
    knobs = (getattr(client, "max_tokens", None), getattr(client, "extra_body", None))
    if getattr(client, "temperature", None) is not None:  # appended, so identities without one are unchanged
        knobs += (client.temperature,)
    return name if knobs == (None, None) else f"{name}+{_sha(*map(repr, knobs))}"


def resolve_description(task_description: str | None, corpus) -> tuple[str | None, str | None]:
    """The sentence the generator and the judge are given. By default the one mteb gives the
    embedding models for this task: `TaskMetadata.prompt` if the task has one, else nothing, which
    judges plain relevance as mteb's own fallback does. `task_description` replaces it."""
    if task_description is not None:
        return task_description, "config:task_description"
    prompt = task_prompt(getattr(corpus.metadata, "prompt", None))
    return (prompt, "mteb:task_prompt") if prompt else (None, None)


def default_cache_folder() -> Path:
    """$MTEB_GYM_CACHE, else ~/.cache/mteb_gym, as mteb uses $MTEB_CACHE, else ~/.cache/mteb."""
    return Path(os.environ.get("MTEB_GYM_CACHE") or Path.home() / ".cache" / "mteb_gym").expanduser()


def _cached_queries(
    path: Path, gen: QueryGenerator, docs: dict[str, str]
) -> tuple[list[Query], int | None, dict | None]:
    """The query set and the generator settings it was written with. The settings are kept with the
    queries, since a later run reads the queries back and never calls the generator itself."""
    if path.exists():
        data = json.loads(path.read_text())
        if data["queries"]:
            return [Query(**q) for q in data["queries"]], data.get("n_generated"), data.get("generator")
        logger.warning("empty query cache at %s (crash artifact); regenerating", path)
    queries = gen.run(docs)
    if not queries:
        raise RuntimeError(
            "query generation kept 0 queries: the generator's answers were not parseable JSON, "
            "or the filter rejected all of them (see the log)"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"generator": gen.settings, "n_generated": gen.n_generated, "queries": [asdict(q) for q in queries]}
    path.write_text(json.dumps(data, indent=2))
    return queries, gen.n_generated, gen.settings


def sample_queries(queries: dict[str, str], n: int, seed: int) -> dict[str, str]:
    """`n` of the dataset's own queries, drawn once with `seed`, in the dataset's order.
    All of them when it has no more than `n`."""
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


def verdict_key(judge: Judge, top_k: int, query_set: str, a: str, rev_a: str | None, b: str, rev_b: str | None) -> str:
    """Identity of a pair's verdicts: the judge and its settings, the resolved prompt, what it is
    shown (top_k documents of doc_chars each), the query set, both models at their revisions."""
    return _verdict_key(_model_id(judge.client), judge.system, top_k, judge.doc_chars, query_set, a, rev_a, b, rev_b)


def _verdict_key(judge_id, system, top_k, doc_chars, query_set, a, rev_a, b, rev_b) -> str:
    return _sha(judge_id, system, top_k, doc_chars, query_set, f"{a}@{rev_a}", f"{b}@{rev_b}")


def verdict_file(vdir: Path, a: str, b: str, key: str) -> Path:
    return vdir / f"{slug(a)}__{slug(b)}-{key}.jsonl"


def cache_files(record: dict, cache_folder: str | Path | None = None) -> dict[str, list[Path]]:
    """The cached files a record was computed from: its query set, if generated, each model's
    prediction file, and one verdict file per model pair. Everything that names them is in the
    record, so publishing a run needs nothing else."""
    cache = Path(cache_folder) if cache_folder is not None else default_cache_folder()
    cfg, task = record["config"], record["task_name"]
    revs = cfg["model_revisions"]
    vdir = cache / "verdicts" / task
    verdicts = [
        verdict_file(
            vdir,
            a,
            b,
            _verdict_key(
                cfg["judge_model"],
                cfg["judge_system"],
                cfg["top_k"],
                cfg["doc_chars"],
                cfg["query_set"],
                a,
                revs[a],
                b,
                revs[b],
            ),
        )
        for a, b in itertools.combinations(cfg["models"], 2)
    ]
    queries = [cache / "queries" / task / f"{cfg['query_set']}.json"] if cfg["arm"] == "synthetic" else []
    predictions = [
        cache / "predictions" / task / f"{slug(m)}@{revs[m]}" / cfg["query_set"] / f"{task}_predictions.json"
        for m in cfg["models"]
    ]
    return {"queries": queries, "predictions": predictions, "verdicts": verdicts}


def pair_subset(n_pairs: int, qids: list[str], k: int | None, seed: int) -> dict[str, set[int]] | None:
    """For each query, the indices of the `k` model pairs to judge, drawn once with `seed`.
    Bradley-Terry ranks from incomplete comparisons, so this cuts judging cost by n_pairs / k."""
    if k is None or k >= n_pairs:
        return None
    rng = random.Random(seed)
    return {qid: set(rng.sample(range(n_pairs), k)) for qid in qids}


def judge_pair_cached(
    vdir: Path, judge: Judge, a: str, b: str, ra: list[Ranked], rb: list[Ranked], key: str
) -> list[Verdict]:
    """Verdicts for one pair under `key`, for the queries in `ra`. Every verdict ever made for the
    pair stays in one JSONL, appended as it is made, so a crash, a rerun, or a run over a subset of
    queries judges only what is missing."""
    jsonl = verdict_file(vdir, a, b, key)
    done: dict[str, Verdict] = {}
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            try:
                v = Verdict(**json.loads(line))
            except (json.JSONDecodeError, TypeError):  # blank, or a line a crash cut short
                continue
            done[v.qid] = v
    todo = [r for r in ra if r.qid not in done]
    if todo:
        if done:
            logger.info("%s vs %s: %d verdicts on disk, judging %d more", a, b, len(done), len(todo))
        vdir.mkdir(parents=True, exist_ok=True)
        if jsonl.exists() and not jsonl.read_bytes().endswith(b"\n"):  # a crash cut the last line short
            with jsonl.open("a", encoding="utf-8") as f:
                f.write("\n")
        lock = threading.Lock()

        def persist(v: Verdict) -> None:
            with lock, jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(v)) + "\n")

        done.update({v.qid: v for v in judge.judge_all(todo, rb, a, b, on_verdict=persist)})
    order = {r.qid: i for i, r in enumerate(ra)}
    return sorted((v for v in done.values() if v.qid in order), key=lambda v: order[v.qid])


class QuerySet(NamedTuple):
    """The queries a run judges on, and how they were obtained."""

    id: str  # part of the prediction and verdict paths
    texts: dict[str, str]
    queries: list[Query] | None  # None when they are the dataset's own
    arm: str  # synthetic | original | own
    n_generated: int | None  # before filtering, for the record
    generator_settings: dict | None


def resolve_queries(corp, cache: Path, queries, gen: QueryGenerator, n_queries: int, seed: int) -> QuerySet:
    """Generate the queries, take the dataset's own, or read the ones you supplied."""
    if queries == "synthetic":
        qid = f"{slug(corp.id)}-{slug(_model_id(gen.client))}-{_sha(sorted(gen.params.items()))}"
        qs, n_generated, settings = _cached_queries(cache / "queries" / corp.name / f"{qid}.json", gen, corp.docs)
        settings = settings or llm_settings(gen.client)  # a query set cached before its settings were kept
        return QuerySet(qid, {q.qid: q.text for q in qs}, qs, "synthetic", n_generated, settings)
    if queries == "original":
        if not corp.queries:
            raise ValueError(f"{corp.name} has no queries of its own")
        texts = sample_queries(corp.queries, n_queries, seed)
        return QuerySet(f"{slug(corp.id)}-original-n{len(texts)}-s{seed}", texts, None, "original", None, None)
    qs = own_queries(queries)
    texts = {q.qid: q.text for q in qs}
    qid = f"{slug(corp.id)}-own-{_sha(*(f'{k}:{v}' for k, v in texts.items()))}"
    return QuerySet(qid, texts, qs, "own", None, None)


def predict(
    corpus: str | Path,
    model: str,
    *,
    generator=None,
    queries="synthetic",
    task_description: str | None = None,
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
    cache = Path(cache_folder) if cache_folder is not None else default_cache_folder()
    corp = corpus_mod.load(corpus)
    description, _ = resolve_description(task_description, corp)
    logger.info("criterion: %s", description or "plain relevance")
    gen = QueryGenerator(
        generator, task_description=description, n_queries=n_queries, seed=seed, filter=filter_queries, workers=workers
    )
    qset = resolve_queries(corp, cache, queries, gen, n_queries, seed)
    logger.info("query set: %s", qset.id)
    revision = mteb.get_model_meta(model).revision
    folder = cache / "predictions" / corp.name / f"{slug(model)}@{revision}" / qset.id
    return retrieval.predict(model, task_mod.build(corp, qset.queries), folder, batch_size=batch_size)


def run(
    corpus: str | Path,
    models: list[str],
    judge,
    generator=None,
    *,
    queries="synthetic",
    task_description: str | None = None,
    n_queries: int = 100,
    top_k: int = 10,
    doc_chars: int = 2000,
    pairs_per_query: int | None = None,
    seed: int = 0,
    filter_queries: bool = True,
    output_folder: str | Path = "results",
    cache_folder: str | Path | None = None,
    batch_size: int = 32,
    workers: int = 8,
) -> Result:
    """Rank `models` on `corpus` with an LLM judge.

    Args:
        corpus: An mteb retrieval task name, a folder of .txt/.md files, or a .jsonl with id and text.
        models: mteb model ids. They run through mteb itself.
        judge: An LLM client (mteb_gym.LLM or MockLLM) that compares the retrieved lists.
        generator: An LLM client that writes the queries. Defaults to the judge.
        queries: "synthetic" (generated), "original" (the dataset's own queries), or your own as a
            .jsonl with id and text, a .txt with one query per line, or a list of strings.
        task_description: One sentence on what counts as a good result, given to generator and judge.
            Defaults to the task's mteb prompt.
        n_queries: Queries to generate.
        top_k: Documents judged per query.
        doc_chars: Characters of each document shown to the judge; longer documents are cut and marked.
        pairs_per_query: Judge only this many random model pairs per query; all pairs by default.
        seed: Seed for document sampling and the bootstrap.
        filter_queries: LLM quality filter and deduplication of generated queries.
        output_folder: Where the record is written, as <task>/<record>.json: the results repository's layout.
        cache_folder: Where queries, predictions and verdicts are cached. Defaults to $MTEB_GYM_CACHE,
            else ~/.cache/mteb_gym.
        batch_size: Encoding batch size.
        workers: Concurrent LLM calls.

    Returns:
        The run's Result: leaderboard, record and path.
    """
    started = time.time()
    out, models = Path(output_folder), list(models)
    cache = Path(cache_folder) if cache_folder is not None else default_cache_folder()
    if not models:
        raise ValueError("no models given")
    gen_client = generator if generator is not None else judge
    if generator is None and queries == "synthetic":
        logger.warning("no generator given: the judge also writes the queries (self-preference risk)")

    corp = corpus_mod.load(corpus)
    description, description_source = resolve_description(task_description, corp)
    logger.info("criterion: %s", description or "plain relevance")
    gen = QueryGenerator(
        gen_client, task_description=description, n_queries=n_queries, seed=seed, filter=filter_queries, workers=workers
    )

    qset = resolve_queries(corp, cache, queries, gen, n_queries, seed)
    logger.info("query set: %s", qset.id)
    query_set, texts, qs, arm = qset.id, qset.texts, qset.queries, qset.arm

    import mteb

    revisions = {m: mteb.get_model_meta(m).revision for m in models}  # mteb's pins: part of every cache identity
    gym_task = task_mod.build(corp, qs)
    labels = task_mod.labels(corp, qs)  # None for your own queries
    ranked: dict[str, list[Ranked]] = {}
    ndcg: dict[str, float] = {}  # the baseline that needs no judge: nDCG@10 against the labels
    for m in models:
        folder = cache / "predictions" / corp.name / f"{slug(m)}@{revisions[m]}" / query_set
        pred = retrieval.predict(m, gym_task, folder, batch_size=batch_size)
        ranked[m] = retrieval.top_k(pred, corp, texts, top_k)
        if labels:
            ndcg[m] = retrieval.ndcg_at_10(pred, labels, corp.ignore_identical_ids)

    # One limit on LLM calls in flight: `workers` pairs at once, each judged one query at a time.
    # Pairs write separate verdict files, so they are independent.
    jd = Judge(judge, instruction=description, workers=1, doc_chars=doc_chars)
    pairs = list(itertools.combinations(models, 2))
    chosen = pair_subset(len(pairs), list(texts), pairs_per_query, seed)  # None: every pair for every query

    def judge_pair(i: int) -> list[Verdict]:
        a, b = pairs[i]
        ra, rb = ranked[a], ranked[b]
        if chosen is not None:
            ra = [r for r in ra if i in chosen[r.qid]]
            rb = [r for r in rb if i in chosen[r.qid]]
        key = verdict_key(jd, top_k, query_set, a, revisions[a], b, revisions[b])
        return judge_pair_cached(cache / "verdicts" / corp.name, jd, a, b, ra, rb, key)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        # map keeps pair order, so the verdict list, and the bootstrap over it, is the same every run
        verdicts = [v for pair in pool.map(judge_pair, range(len(pairs))) for v in pair]
    logger.info("judged %d pairs", len(pairs))

    ratings = rate(verdicts, seed=seed)
    experiment = {
        "corpus_id": corp.id,
        "query_set": query_set,
        "arm": arm,
        "judge_model": _model_id(judge),
        "generator_model": _model_id(gen_client) if arm == "synthetic" else None,
        "task_description": description,
        "task_description_source": description_source,
        "judge_system": jd.system,
        "n_queries_generated": qset.n_generated,
        "n_queries": len(texts),
        "top_k": top_k,
        "doc_chars": doc_chars,
        "pairs_per_query": pairs_per_query,
        "seed": seed,
        **({f"gen_{k}": v for k, v in gen.params.items()} if arm == "synthetic" else {}),
        "models": models,
        "model_revisions": revisions,
    }
    experiment["config_hash"] = results.config_hash(experiment)
    path = results.record_path(out, corp.name, experiment)
    if path.exists():  # same configuration, same verdicts: the record stands, agreement included
        return Result.from_disk(path)
    llms = {"judge": llm_settings(judge), "generator": qset.generator_settings}
    label_source = "dataset" if arm == "original" else "seed_documents" if labels else None
    result = Result(
        results.build_record(
            corp, experiment, ratings, verdicts, time.time() - started, revisions, llms, ndcg=ndcg, labels=label_source
        ),
        path,
    )
    result.to_disk()
    return result
