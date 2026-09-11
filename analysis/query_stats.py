"""
Descriptive statistics of a run's synthetic queries, next to the corpus's own queries.

    python -m analysis.query_stats --output-folder results --record results/records/<synthetic record>.json \
        --original-queries original.json --docs docs.json --out query_stats.json

The ranking is only as good as the queries it was judged on, and the generator's prompt asks for
something specific: one natural query per sample of documents, answerable from the corpus but not a
restatement of any shown document. This script measures the cheap proxies for whether that happened,
on the queries the run kept, and puts the corpus's human queries beside them when they are given:

    count, mean and median length in words
    question share       ends with "?" or starts with a question word or auxiliary (QUESTION_STARTS)
    copied-word share    content words of a query (stopwords removed) that occur in its seed documents;
                         needs the corpus text (--docs), and is reported as null without it, never as 0;
                         the corpus's own queries have no seed documents, so it is null for them too
    quality scores       the generator's own 1-5 filter score per kept query, from the queries file; the
                         filter records 3 for an answer it could not parse, and the file does not say which
    filter accounting    n_generated in the queries file against the kept count: dropped and drop rate; the
                         record's n_queries_generated must agree with the file or the script refuses
    coverage             which kept qids the run's prediction files and each pair's verdict file contain

The package does not store why a query was dropped. n_generated is the count that passed the
length and degeneracy heuristics during generation; the drops after it are the quality gate, the
near-duplicate removal and the cut to n_queries taken together, and the output says so instead of
splitting them into categories that were never recorded.

The queries file is out/queries/<query_set>.json as run() wrote it. The record's prediction and
verdict files are located through mteb_gym.reliability only, to check that the queries the run
retrieved and judged on are the ones in that file.

Word counts split on whitespace. Copied-word share tokenises to lowercase [a-z0-9]+ runs, drops
STOPWORDS, and counts a query word once however many times it occurs.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from mteb_gym import reliability as rel
from mteb_gym.results import Result

QUESTION_STARTS = frozenset(
    "what which who whom whose when where why how is are do does did can could should would will".split()
)

STOPWORDS = frozenset(
    """
    a an the and or but nor so yet for of in on at to from by with without about as into onto over under
    between among through during before after above below up down out off again further then once here
    there where when why how all any both each few more most other some such no not only own same than too
    very s t can will just don should now is are was were be been being have has had having do does did
    doing i me my myself we our ours ourselves you your yours yourself yourselves he him his himself she
    her hers herself it its itself they them their theirs themselves what which who whom whose this that
    these those am if because until while against also may might must shall would could
    """.split()
)

FILTER_NOTE = (
    "n_generated counts queries that passed the length and degeneracy heuristics during generation; the "
    "package does not record why a query was dropped after that, so the drop count folds together the "
    "quality gate (score < gen_min_score), near-duplicate removal (Jaccard >= gen_dedup) and the cut to "
    "n_queries. Heuristic rejections during generation are not counted anywhere."
)

QUALITY_NOTE = (
    "the quality filter writes 3 when it cannot parse the scoring model's answer (QueryGenerator._llm_quality), "
    "so a 3 in the queries file may be a defaulted score rather than a judged one; the file does not say which."
)

_TOKEN = re.compile(r"[a-z0-9]+")


# ----------------------------------------------------------------------------- per-query measures
def word_count(text: str) -> int:
    return len(text.split())


def is_question(text: str) -> bool:
    words = text.strip().lower().split()
    if not words:
        return False
    first = _TOKEN.findall(words[0])
    return text.strip().endswith("?") or bool(first and first[0] in QUESTION_STARTS)


def content_words(text: str) -> set[str]:
    """Lowercase alphanumeric tokens with stopwords removed."""
    return {w for w in _TOKEN.findall(text.lower()) if w not in STOPWORDS}


def copied_counts(text: str, seed_texts: Iterable[str]) -> tuple[int, int]:
    """(content words of the query found in any seed document, content words of the query)."""
    words = content_words(text)
    if not words:
        return 0, 0
    seen: set[str] = set()
    for doc in seed_texts:
        seen |= content_words(doc)
    return len(words & seen), len(words)


def copied_share(text: str, seed_texts: Iterable[str]) -> float | None:
    """Share of a query's content words found in any of its seed documents; None without content words."""
    matched, total = copied_counts(text, seed_texts)
    return matched / total if total else None


# ----------------------------------------------------------------------------- aggregates
def _check_template(text: str, source: str) -> None:
    if "${" in text:
        raise ValueError(f"{source} contains template-looking text ({text[:60]!r}); refusing to analyse it")


def describe(texts: Mapping[str, str]) -> dict[str, Any]:
    """count, length in words and question share over {qid: text}."""
    lengths = [word_count(t) for t in texts.values()]
    return {
        "count": len(lengths),
        "mean_words": statistics.fmean(lengths) if lengths else None,
        "median_words": statistics.median(lengths) if lengths else None,
        "question_share": sum(is_question(t) for t in texts.values()) / len(lengths) if lengths else None,
    }


def copied_word_share(queries: Sequence[Mapping[str, Any]], docs: Mapping[str, str] | None) -> dict[str, Any]:
    """Mean per-query and pooled copied-word share over kept queries whose seed documents are all in `docs`."""
    if docs is None:
        return {"mean_per_query": None, "pooled": None, "n_scored": 0, "n_missing_seed_docs": None, "n_no_content": 0}
    shares: list[float] = []
    missing = no_content = 0
    matched = total = 0
    for q in queries:
        seeds = q.get("seed_doc_ids") or []
        if not seeds or any(d not in docs for d in seeds):
            missing += 1
            continue
        m, t = copied_counts(q["text"], (docs[d] for d in seeds))
        if not t:
            no_content += 1
            continue
        shares.append(m / t)
        matched += m
        total += t
    return {
        "mean_per_query": statistics.fmean(shares) if shares else None,
        "pooled": matched / total if total else None,
        "n_scored": len(shares),
        "n_missing_seed_docs": missing,
        "n_no_content": no_content,
    }


def quality_distribution(queries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scores = [q["quality"] for q in queries if q.get("quality") is not None]
    counts: dict[str, int] = {}
    for s in sorted(scores):
        counts[str(s)] = counts.get(str(s), 0) + 1
    return {
        "counts": counts,
        "mean": statistics.fmean(scores) if scores else None,
        "n_scored": len(scores),
        "n_unscored": len(queries) - len(scores),
        "note": QUALITY_NOTE,
    }


def filter_accounting(n_generated: int | None, n_kept: int, config: Mapping[str, Any]) -> dict[str, Any]:
    """`n_generated` is the queries file's count; the record's n_queries_generated must agree with it."""
    in_record = config.get("n_queries_generated")
    if n_generated is not None and in_record is not None and in_record != n_generated:
        raise ValueError(
            f"the queries file says n_generated={n_generated} but the record says n_queries_generated={in_record}: "
            "the queries file was rewritten after the run and no longer describes the queries the run judged"
        )
    dropped = None if n_generated is None else n_generated - n_kept
    return {
        "n_generated": n_generated,
        "n_generated_in_record": in_record,
        "n_kept": n_kept,
        "n_dropped": dropped,
        "drop_rate": dropped / n_generated if n_generated else None,
        "n_queries_requested": config.get("n_queries"),
        "filter_on": config.get("gen_filter"),
        "min_score": config.get("gen_min_score"),
        "dedup": config.get("gen_dedup"),
        "note": FILTER_NOTE,
    }


# ----------------------------------------------------------------------------- the run's artifacts
def queries_file(out: Path, record: Mapping[str, Any]) -> Path:
    return out / "queries" / f"{record['config']['query_set']}.json"


def load_queries(out: Path, record: Mapping[str, Any]) -> tuple[list[dict], int | None]:
    data = json.loads(queries_file(out, record).read_text())
    for q in data["queries"]:
        _check_template(q["text"], f"query {q.get('qid')}")
    return data["queries"], data.get("n_generated")


def _as_text(what: str, key: str, value: Any) -> str:
    """A string as is; an mteb corpus row ({title, text}) joined the way mteb_gym.corpus does; anything else refused."""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping) and ("text" in value or "title" in value):
        return ((value.get("title") or "") + " " + (value.get("text") or "")).strip()
    raise ValueError(f"{what} {key!r} is a {type(value).__name__}, not text or a {{title, text}} row")


def load_texts(path: str | Path, what: str) -> dict[str, str]:
    """{id: text} from a JSON file of id -> text (or id -> {title, text} row)."""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{what} at {path} must be a JSON object of id -> text")
    texts = {str(k): _as_text(what, str(k), v) for k, v in data.items()}
    if what == "original queries":
        for k, v in texts.items():
            _check_template(v, f"original query {k}")
    return texts


def judged_coverage(out: Path, record: Mapping[str, Any], kept_qids: set[str]) -> dict[str, Any]:
    """Which of the kept qids the run's prediction and verdict files actually cover; missing files are listed.

    Pairs are taken in the order run() judged them (config["models"], itertools.combinations); a pair stored
    the other way round is still found. n_kept_not_judged is over the union of all pairs' verdicts, so
    n_kept_not_judged_in_some_pair is the one that catches a pair that stopped early."""
    models = list(record["config"].get("models") or [r["model"] for r in record["ratings"]])
    predicted: set[str] = set()
    missing: list[str] = []
    for m in models:
        path = rel.prediction_file(out, record, m)
        if not path.exists():
            missing.append(str(path))
            continue
        predicted |= set(json.loads(path.read_text())["default"]["test"])
    judged: set[str] = set()
    unjudged_in_some_pair: set[str] = set()
    pairs_found = 0
    for i, a in enumerate(models):
        for b in models[i + 1 :]:
            path = rel.verdict_file(out, record, a, b)
            if not (path.exists() or path.with_suffix(".jsonl").exists()):
                path = rel.verdict_file(out, record, b, a)
            try:
                qids = set(rel.load_verdicts(path))
            except FileNotFoundError:
                missing.append(str(rel.verdict_file(out, record, a, b)))
                continue
            pairs_found += 1
            judged |= qids
            unjudged_in_some_pair |= kept_qids - qids
    return {
        "n_models": len(models),
        "n_pairs_expected": len(models) * (len(models) - 1) // 2,
        "n_pairs_found": pairs_found,
        "n_qids_predicted": len(predicted),
        "n_qids_judged": len(judged),
        "n_kept_not_predicted": len(kept_qids - predicted),
        "n_kept_not_judged": len(kept_qids - judged),
        "n_kept_not_judged_in_some_pair": len(unjudged_in_some_pair),
        "n_judged_not_kept": len(judged - kept_qids),
        "missing_files": missing,
    }


# ----------------------------------------------------------------------------- entry points
def query_stats(
    output_folder: str | Path,
    record: str | Path,
    *,
    original_queries: Mapping[str, str] | None = None,
    docs: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """The readout for one synthetic-arm record; `original_queries` and `docs` are optional {id: text}."""
    out = Path(output_folder)
    rec = Result.from_disk(record).record
    c = rec["config"]
    if c.get("arm") != "synthetic":
        raise ValueError(f"arm is {c.get('arm')!r}: query statistics need the synthetic-arm record")
    queries, n_generated = load_queries(out, rec)
    texts = {q["qid"]: q["text"] for q in queries}
    synthetic = {
        **describe(texts),
        "copied_word_share": copied_word_share(queries, docs),
        "quality": quality_distribution(queries),
        "filter": filter_accounting(n_generated, len(queries), c),
        "coverage": judged_coverage(out, rec, set(texts)),
    }
    original = None
    if original_queries is not None:
        original = {
            **describe(original_queries),
            "copied_word_share": {"mean_per_query": None, "pooled": None, "note": "no seed documents"},
        }
    return {
        "task_name": rec["task_name"],
        "record": str(Path(record)),
        "query_set": c["query_set"],
        "generator_model": c.get("generator_model"),
        "config_hash": c.get("config_hash"),
        "docs_given": docs is not None,
        "synthetic": synthetic,
        "original": original,
        "notes": [FILTER_NOTE, QUALITY_NOTE, "copied-word share is null when --docs is not given, not 0."],
    }


def _fmt(x: Any) -> str:
    if x is None:
        return "null"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def markdown_table(stats: Mapping[str, Any]) -> str:
    s, o = stats["synthetic"], stats.get("original") or {}
    rows = [
        ("queries", s["count"], o.get("count")),
        ("mean length (words)", s["mean_words"], o.get("mean_words")),
        ("median length (words)", s["median_words"], o.get("median_words")),
        ("question share", s["question_share"], o.get("question_share")),
        ("copied-word share, mean per query", s["copied_word_share"]["mean_per_query"], None),
        ("copied-word share, pooled", s["copied_word_share"]["pooled"], None),
        ("quality score, mean", s["quality"]["mean"], None),
        ("quality score counts", json.dumps(s["quality"]["counts"]), None),
        ("n_generated", s["filter"]["n_generated"], None),
        ("n_dropped", s["filter"]["n_dropped"], None),
        ("drop rate", s["filter"]["drop_rate"], None),
    ]
    head = f"| metric | synthetic ({stats['generator_model']}) | original |"
    lines = [head, "|---|---|---|"] + [f"| {name} | {_fmt(a)} | {_fmt(b)} |" for name, a, b in rows]
    if stats.get("original") is None:
        lines += ["", "original: not given (--original-queries)."]
    return "\n".join(lines)


def dump_original_queries(task_name: str, path: str | Path) -> Path:
    """Write the corpus's own queries as {qid: text} JSON for --original-queries. Needs mteb."""
    from mteb_gym.corpus import load

    corp = load(task_name)
    if not corp.queries:
        raise ValueError(f"{task_name} has no queries of its own")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(corp.queries, indent=1))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--output-folder", default="results", help="the gym's output folder (queries/ inside)")
    ap.add_argument("--record", required=True, help="a synthetic-arm record under <output-folder>/records/")
    ap.add_argument("--original-queries", default=None, help="JSON {qid: text} of the corpus's own queries")
    ap.add_argument("--docs", default=None, help="JSON {docid: text} of the corpus, for the copied-word share")
    ap.add_argument("--out", default="query_stats.json", help="JSON readout; the markdown table goes next to it")
    ap.add_argument("--dump-original", metavar="TASK", default=None, help="write TASK's queries to --original-queries")
    args = ap.parse_args(argv)
    if args.dump_original:
        if not args.original_queries:
            ap.error("--dump-original needs --original-queries as the destination")
        print(f"wrote {dump_original_queries(args.dump_original, args.original_queries)}")
    original = load_texts(args.original_queries, "original queries") if args.original_queries else None
    docs = load_texts(args.docs, "docs") if args.docs else None
    stats = query_stats(args.output_folder, args.record, original_queries=original, docs=docs)
    table = markdown_table(stats)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=1))
    out.with_suffix(".md").write_text(table + "\n")
    print(table)
    print(f"\n{FILTER_NOTE}\n-> {out} and {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
