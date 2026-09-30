"""
The result of a run: one record with ratings, the resolved configuration, judge
diagnostics and versions, so a reported ranking can be reproduced offline.
`Result` wraps one record, `Results` a directory of them; both give a dataframe
and agreement with official MTEB scores.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .rank import format_leaderboard


def config_hash(config: dict[str, Any], length: int = 8) -> str:
    """Stable short hash over the experiment-defining config (never over itself)."""
    payload = {k: v for k, v in config.items() if k != "config_hash"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()[
        :length
    ]


def _version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _short(model: str | None) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", (model or "").split("/")[-1]).strip("-")


def record_path(out: Path, task_name: str, experiment: dict[str, Any]) -> Path:
    """out/<task>/<task>__<judge>__<generator or arm>__q<n>-s<seed>-<hash>.json, the results repository's layout"""
    second = (
        _short(experiment["generator_model"]) if experiment["arm"] == "synthetic" else f"{experiment['arm']}-queries"
    )
    name = f"{task_name}__{_short(experiment['judge_model'])}__{second}__q{experiment['n_queries']}-s{experiment['seed']}-{experiment['config_hash']}.json"
    return Path(out) / task_name / name


def verdict_diagnostics(verdicts: list[Any]) -> dict[str, Any]:
    """Judge diagnostics from persisted verdicts, so cached and resumed runs report the truth."""
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
        "commit_rate": (n - ties) / n if n else None,
        "tie_rate": ties / n if n else None,
        "a_first_rate": first / decisive if decisive else None,
        "parse_failure_rate": failures / asks if asks else None,
        "identical_retrieval_rate": identical / n if n else None,
    }


def build_record(
    corpus,
    experiment,
    ratings,
    verdicts,
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
        "diagnostics": verdict_diagnostics(verdicts),
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
    }  # agreement() adds "agreement"


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
        self.path.write_text(json.dumps(self.record, indent=2))
        return self.path

    @property
    def leaderboard(self) -> str:
        return format_leaderboard(self.record["ratings"])

    def to_dataframe(self):
        """One row per model: task, judge, generator, model, revision, rating, ci_low, ci_high, n_queries."""
        import pandas as pd

        c = self.record["config"]
        head = {
            "task": self.record["task_name"],
            "judge": c["judge_model"],
            "generator": c["generator_model"],
            "arm": c["arm"],
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


def load_results(root: str | Path) -> Results:
    """All records under `root`: a results folder's or results repository's <task>/ folders, or an
    older output folder's records/."""
    paths = sorted(p for p in Path(root).rglob("*.json") if p.parent.name in ("records", p.name.split("__")[0]))
    return Results([Result.from_disk(p) for p in paths])
