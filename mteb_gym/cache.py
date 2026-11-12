"""
Every stage a run can reuse (default ~/.cache/mteb_gym, or $MTEB_GYM_CACHE), laid out like the dataset its files are
published to.

    <cache_folder>/queries/<task>/<query set>.json                       generated queries and their generator
    <cache_folder>/predictions/<task>/<model>@<revision>/<query set>/    mteb's prediction file
    <cache_folder>/grades/<task>/<judge>-<key>.jsonl                     pointwise: one line per graded document
    <cache_folder>/verdicts/<task>/<pair>-<key>.jsonl                    pairwise: one line per comparison
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import os
import threading
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Never, TypeVar

from tqdm.auto import tqdm

from .judge import Grade, PairwiseJudge, PointwiseJudge, Verdict
from .local import load_now
from .retrieval import Ranked, slug

_JudgmentT = TypeVar("_JudgmentT", Grade, Verdict)

logger = logging.getLogger(__name__)


def sha(*parts: object) -> str:
    canonical = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def default_cache_folder() -> Path:
    """$MTEB_GYM_CACHE, else ~/.cache/mteb_gym."""
    return Path(os.environ.get("MTEB_GYM_CACHE") or Path.home() / ".cache" / "mteb_gym").expanduser()


def resolve_cache_folder(cache_folder: str | Path | None) -> Path:
    return Path(cache_folder) if cache_folder is not None else default_cache_folder()


def model_id(client: object) -> str:
    """The judge or generator as its answers depend on it: the model, and the temperature, output cap and server
    settings it was built with."""
    name = str(getattr(client, "model", type(client).__name__))
    knobs = [getattr(client, k, None) for k in ("max_tokens", "extra_body", "temperature")]
    return name if not any(k is not None for k in knobs) else f"{name}+{sha(*knobs)}"


def texts_hash(texts: dict[str, str]) -> str:
    return sha(sorted(texts.items()))


def capped(revision: str | None, max_seq_length: int | None) -> str | None:
    """A model's identity in the caches: its revision, and its length cap when it has one, as a cap changes what it
    retrieves."""
    return revision if max_seq_length is None else f"{revision}-max{max_seq_length}"


def prediction_folder(
    cache: Path, task: str, model: str, revision: str | None, query_set: str, max_seq_length: int | None = None
) -> Path:
    return cache / "predictions" / task / f"{slug(model)}@{capped(revision, max_seq_length)}" / query_set


def verdict_key(
    judge_id: str,
    system: str,
    top_k: int,
    doc_chars: int,
    query_set: str,
    a: str,
    rev_a: str | None,
    b: str,
    rev_b: str | None,
) -> str:
    """Identity of a pair's verdicts: the judge, what it is told and shown, the query set, both models at their
    revisions."""
    return sha(judge_id, system, top_k, doc_chars, query_set, f"{a}@{rev_a}", f"{b}@{rev_b}")


def grades_key(judge_id: str, system: str, instruction: str | None, doc_chars: int, query_set: str) -> str:
    """Identity of a query set's grades. The pool's depth is not part of it: a grade belongs to a (query, document), so
    a deeper pool only adds grades."""
    return sha(judge_id, system, instruction, doc_chars, query_set)


def verdict_file(folder: Path, a: str, b: str, key: str) -> Path:
    return folder / f"{slug(a)}__{slug(b)}-{key}.jsonl"


def grade_file(folder: Path, judge_id: str, key: str) -> Path:
    return folder / f"{slug(judge_id)}-{key}.jsonl"


def cache_files(record: dict[str, Any], cache_folder: str | Path | None = None) -> dict[str, list[Path]]:
    """The cached files a record was computed from, named from the record alone."""
    cache = resolve_cache_folder(cache_folder)
    cfg, task = record["config"], record["task_name"]
    revs, cap = cfg["model_revisions"], cfg.get("max_seq_length")
    if cfg["judging"] == "pointwise":
        key = grades_key(
            cfg["judge_model"], cfg["judge_system"], cfg["task_instruction"], cfg["doc_chars"], cfg["query_set"]
        )
        judgments = [grade_file(cache / "grades" / task, cfg["judge_model"], key)]
    else:
        judgments = [
            verdict_file(
                cache / "verdicts" / task,
                a,
                b,
                verdict_key(
                    cfg["judge_model"],
                    cfg["judge_system"],
                    cfg["top_k"],
                    cfg["doc_chars"],
                    cfg["query_set"],
                    a,
                    capped(revs[a], cap),
                    b,
                    capped(revs[b], cap),
                ),
            )
            for a, b in itertools.combinations(cfg["models"], 2)
        ]
    recipe = cfg["query_set"].rsplit("-", 1)[0]  # the generated set's file is named before its queries exist
    if cfg["pairs_per_query"] is not None:  # a random subset of pairs leaves the others without a file
        judgments = [p for p in judgments if p.exists()]
    queries = [cache / "queries" / task / f"{recipe}.json"] if cfg["query_source"] == "synthetic" else []
    predictions = [
        prediction_folder(cache, task, m, revs[m], cfg["query_set"], cap) / f"{task}_predictions.json"
        for m in cfg["models"]
    ]
    return {"queries": queries, "predictions": predictions, "judgments": judgments}


def _read_jsonl(path: Path, cls: type[_JudgmentT]) -> list[_JudgmentT]:
    if not path.exists():
        return []
    items = []
    for line in path.read_text().splitlines():
        try:
            items.append(cls(**json.loads(line)))
        except (json.JSONDecodeError, TypeError):  # blank, or a line a crash cut short
            continue
    return items


def _appender(path: Path, run: dict[str, Any]) -> Callable[[Grade | Verdict], None]:
    """A function that appends an item to `path`, so a crash loses at most the item being written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not path.read_bytes().endswith(b"\n"):  # a crash cut the last line short
        with path.open("a", encoding="utf-8") as f:
            f.write("\n")
    lock = threading.Lock()

    def append(item: Grade | Verdict) -> None:
        with lock, path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({**asdict(item), **run}) + "\n")

    return append


def judge_pool_cached(
    folder: Path,
    judge: PointwiseJudge,
    ranked: dict[str, list[Ranked]],
    docs: dict[str, str],
    key: str,
    query_set: str = "",
    bar: tqdm[Never] | None = None,
    source_pairs: dict[str, list[str]] | None = None,
) -> list[Grade]:
    """A grade for every document in every query's pool (the union of the models' lists). A crash, a rerun or an added
    model grades only what is missing, and a failed answer counts as missing."""
    path = grade_file(folder, model_id(judge.client), key)
    done = {(g.qid, g.doc_id): g for g in _read_jsonl(path, Grade) if g.parsed_ok}
    query = {r.qid: r.query for lists in ranked.values() for r in lists}
    pool = sorted({(r.qid, d) for lists in ranked.values() for r in lists for d in r.doc_ids})
    pool = sorted(set(pool) | {(q, d) for q, ids in (source_pairs or {}).items() if q in query for d in ids})
    todo = [(q, query[q], d, docs[d]) for q, d in pool if (q, d) not in done]
    if todo:
        load_now(judge.client)
    if bar is not None:  # after loading, so the bar times only the grading
        bar.reset(total=len(pool))
        bar.update(len(pool) - len(todo))
    if todo:
        logger.info("grading %d documents (%d already graded)", len(todo), len(pool) - len(todo))
        run = dict(task=folder.name, judge=model_id(judge.client), query_set=query_set)
        write = _appender(path, run)

        def append(grade: Grade) -> None:
            write(grade)
            if bar is not None and grade.parsed_ok:
                bar.update()

        grades = judge.grade_all(todo, on_grade=append)
        failed = {(g.qid, g.doc_id) for g in grades if not g.parsed_ok}
        if failed:
            logger.info("asking again for %d failed grades", len(failed))
            grades += judge.grade_all([t for t in todo if (t[0], t[2]) in failed], on_grade=append)
        done.update(
            {
                (g.qid, g.doc_id): replace(g, task=run["task"], judge=run["judge"], query_set=run["query_set"])
                for g in grades
            }
        )
    return [done[p] for p in pool]


def verdicts_done(path: Path) -> dict[str, Verdict]:
    """The usable verdicts on disk, by query."""
    return {v.qid: v for v in _read_jsonl(path, Verdict) if all(v.parsed_ok)}


def judge_pair_cached(
    folder: Path,
    judge: PairwiseJudge,
    a: str,
    b: str,
    ra: list[Ranked],
    rb: list[Ranked],
    key: str,
    query_set: str = "",
    bar: tqdm[Never] | None = None,
) -> list[Verdict]:
    """Verdicts for one pair under `key`, for the queries in `ra`. A crash, a rerun or a run over a subset of queries
    judges only what is missing, and a failed answer counts as missing."""
    path = verdict_file(folder, a, b, key)
    done = verdicts_done(path)
    todo = [r for r in ra if r.qid not in done]
    if bar is not None:
        bar.update(len(ra) - len(todo))
    if todo:
        if done:
            logger.info("%s vs %s: %d verdicts on disk, judging %d more", a, b, len(done), len(todo))
        run = dict(task=folder.name, judge=model_id(judge.client), query_set=query_set)
        write = _appender(path, run)

        def append(verdict: Verdict) -> None:
            write(verdict)
            if bar is not None:
                bar.update()

        verdicts = judge.judge_all(todo, rb, a, b, on_verdict=append)
        done.update(
            {v.qid: replace(v, task=run["task"], judge=run["judge"], query_set=run["query_set"]) for v in verdicts}
        )
    order = {r.qid: i for i, r in enumerate(ra)}
    return sorted((v for v in done.values() if v.qid in order), key=lambda v: order[v.qid])
