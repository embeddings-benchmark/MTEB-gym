"""
The result of a run: one record with ratings, the resolved configuration, judge
diagnostics and versions, so a reported ranking can be reproduced offline.
`Result` wraps one record, `Results` a directory of them; both give a dataframe
and agreement with official MTEB scores.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .rank import format_leaderboard


def config_hash(config: dict[str, Any], length: int = 8) -> str:
    """A short, stable hash of the configuration."""
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()[
        :length
    ]


def _version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def git_revision() -> str | None:
    """The gym's commit when it runs from a git checkout. An installed package has none: the folder above it may be
    the user's own repository."""
    root = Path(__file__).resolve().parents[1]
    if not (root / ".git").exists():
        return None
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _short(model: str | None) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", (model or "").split("/")[-1]).strip("-")


def record_path(out: Path, task_name: str, experiment: dict[str, Any]) -> Path:
    """out/<task>/<task>__<judge>__<generator or query_source>__q<n>-s<seed>-<hash>.json, the results repository's layout"""
    second = (
        _short(experiment["generator_model"])
        if experiment["query_source"] == "synthetic"
        else f"{experiment['query_source']}-queries"
    )
    name = f"{task_name}__{_short(experiment['judge_model'])}__{second}__q{experiment['n_queries']}-s{experiment['seed']}-{experiment['config_hash']}.json"
    return Path(out) / task_name / name


def verdict_diagnostics(verdicts: list[Any]) -> dict[str, Any]:
    """Judge diagnostics from the verdicts, including any read back from the cache."""
    n = len(verdicts)
    identical = sum(v.raw == ["identical"] for v in verdicts)
    ties = sum(float(v.score_a) == 0.5 for v in verdicts)
    asks = failures = first = decisive = 0
    for v in verdicts:
        asks += len(v.parsed_ok)
        failures += sum(not ok for ok in v.parsed_ok)
        for w in v.raw:
            if w in ("A", "B"):
                decisive += 1
                first += w == "A"
    return {
        "judge_calls": asks,
        "n_comparisons": n,
        "n_shortened": sum(getattr(v, "doc_chars", None) is not None for v in verdicts),
        "commit_rate": (n - ties) / n if n else None,
        "tie_rate": ties / n if n else None,
        "a_first_rate": first / decisive if decisive else None,
        "parse_failure_rate": failures / asks if asks else None,
        "identical_retrieval_rate": identical / n if n else None,
    }


def grade_diagnostics(grades: list[Any]) -> dict[str, Any]:
    n = len(grades)
    failed = sum(not g.parsed_ok for g in grades)
    ok = [g.grade for g in grades if g.parsed_ok]
    n_queries = len({g.qid for g in grades})
    return {
        "n_grades": n,
        "n_shortened": sum(getattr(g, "doc_chars", None) is not None for g in grades),
        "pool_size_mean": n / n_queries if n_queries else None,
        "grade_mean": sum(ok) / len(ok) if ok else None,
        "parse_failure_rate": failed / n if n else None,
    }


def build_record(
    corpus,
    experiment,
    ratings,
    diagnostics: dict,
    evaluation_time: float,
    revisions: dict[str, str | None],
    llms: dict | None = None,
    ndcg: dict[str, float] | None = None,
    labels: str | None = None,
) -> dict:
    dataset = getattr(corpus.metadata, "dataset", None) or {}
    return {
        "task_name": corpus.name,
        "source": corpus.source,
        "corpus_id": corpus.id,
        "dataset": {"path": dataset.get("path"), "revision": dataset.get("revision")},
        "mteb_version": _version("mteb"),
        "gym_version": _version("mteb-gym"),
        "gym_revision": git_revision(),
        "evaluation_time": float(evaluation_time),
        "config": experiment,
        "llms": llms,  # what the judge and generator actually ran with; informational, not identity
        "labels": labels,  # what ndcg_at_10 below is scored against: "seed_documents", "dataset", or None
        "diagnostics": diagnostics,
        "ratings": [
            {
                "model": m.name,
                "revision": revisions.get(m.name),
                "rating": m.rating,
                "ci_low": m.ci_low,
                "ci_high": m.ci_high,
                "wins": m.wins,
                "losses": m.losses,
                "ties": m.ties,
                "n": m.n,
                "ndcg_at_10": (ndcg or {}).get(m.name),  # the no-judge baseline
            }
            for m in ratings
        ],
    }


@dataclass
class Result:
    """One run's record."""

    record: dict
    path: Path | None = None

    @classmethod
    def from_disk(cls, path: str | Path) -> Result:
        return cls(json.loads(Path(path).read_text()), Path(path))

    def to_disk(self, path: str | Path | None = None) -> Path:
        self.path = Path(path or self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f".tmp{os.getpid()}")
        temporary.write_text(json.dumps(self.record, indent=2))
        temporary.replace(self.path)
        return self.path

    @property
    def leaderboard(self) -> str:
        return format_leaderboard(self.record["ratings"])

    def to_dataframe(self):
        """One row per model, with the task, judge, generator and query source."""
        import pandas as pd

        c = self.record["config"]
        head = {
            "task": self.record["task_name"],
            "judge": c["judge_model"],
            "generator": c["generator_model"],
            "query_source": c["query_source"],
            "n_queries": c["n_queries"],
        }
        return pd.DataFrame([{**head, **r} for r in self.record["ratings"]])

    def agreement(self, *, evaluate_missing: bool = False, bootstrap: int = 1000, seed: int = 0) -> dict:
        """Rank agreement with the official MTEB ranking; written into the record as "agreement"."""
        from . import agreement as agree

        if self.record["source"] != "mteb":
            agreement = {"error": "local corpus: no official scores to compare with"}
        else:
            ratings = {r["model"]: r["rating"] for r in self.record["ratings"]}
            truth, source = agree.fetch_truth(
                list(ratings), self.record["task_name"], evaluate_missing=evaluate_missing
            )
            agreement = agree.correlate(ratings, truth, bootstrap=bootstrap, seed=seed)
            agreement["truth_source"] = source
            ndcg = {r["model"]: r["ndcg_at_10"] for r in self.record["ratings"] if r.get("ndcg_at_10") is not None}
            if ndcg:  # the same comparison for the ranking by nDCG against the run's labels, no judge involved
                agreement["labels_baseline"] = {
                    "labels": self.record.get("labels"),
                    **agree.correlate(ndcg, truth, bootstrap=bootstrap, seed=seed),
                }
        self.record["agreement"] = agreement
        if self.path:
            self.to_disk()
        return agreement


@dataclass
class Results:
    """Every record under a directory."""

    results: list[Result]

    def to_dataframe(self):
        import pandas as pd

        return pd.concat([r.to_dataframe() for r in self.results], ignore_index=True)

    def agreement(self, **kwargs) -> dict[str, dict]:
        return {str(r.path): r.agreement(**kwargs) for r in self.results}

    def get_benchmark_result(self, benchmark):
        """Each model's rank and mean rating on `benchmark`, a `Benchmark` or its name, as a DataFrame, as mteb's
        `get_benchmark_result` gives."""
        from .benchmarks import get_benchmark

        benchmark = get_benchmark(benchmark) if isinstance(benchmark, str) else benchmark
        return benchmark._create_summary_table(self)


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def download_results(
    results_repo: str | None = None,
    download_latest: bool = True,
    revision: str | None = None,
    cache_folder: str | Path | None = None,
) -> Path:
    """A shallow clone of the results repository in the cache, as mteb keeps its results. An existing clone is reused,
    and moved to the repository's latest commit with `download_latest`, or to `revision`."""
    from .cache import resolve_cache_folder
    from .submit import RESULTS_REPOSITORY

    results_repo = results_repo or RESULTS_REPOSITORY
    clone = resolve_cache_folder(cache_folder) / "results_repo"
    if not clone.exists():
        clone.parent.mkdir(parents=True, exist_ok=True)
        _git("clone", "--quiet", "--depth", "1", results_repo, str(clone), cwd=clone.parent)
        update = revision is not None
    elif (origin := _git("config", "--get", "remote.origin.url", cwd=clone)) != results_repo:
        raise ValueError(f"{clone} is a clone of {origin}, not {results_repo}")
    else:
        update = download_latest or revision is not None
    if update:
        _git("fetch", "--quiet", "--depth", "1", "origin", revision or "HEAD", cwd=clone)
        _git("reset", "--quiet", "--hard", "FETCH_HEAD", cwd=clone)
    return clone


def load_results(
    root: str | Path | None = None,
    *,
    results_repo: str | None = None,
    download_latest: bool = True,
    revision: str | None = None,
) -> Results:
    """All records under `root`: a results folder's or results repository's <task>/ folders. Without `root`, the
    records of the results repository, fetched into the cache as `mteb.load_results` fetches its results."""
    if root is None:
        root = download_results(results_repo, download_latest, revision) / "results"
    paths = sorted(p for p in Path(root).rglob("*.json") if p.parent.name == p.name.split("__")[0])
    return Results([Result.from_disk(p) for p in paths])
