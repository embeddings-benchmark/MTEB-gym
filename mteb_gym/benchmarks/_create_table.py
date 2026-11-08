"""Leaderboard tables from records: one record per task of a benchmark."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from ..results import Results
    from .benchmark import Benchmark


def benchmark_records(results: Results, benchmark: Benchmark) -> dict[str, dict]:
    """The record of each task that has one. Where a task has several, the one with the most queries; a tie is an
    error, because the table would depend on which was read first."""
    found: dict[str, list[dict]] = {}
    for result in results.results:
        if benchmark.matches(result.record):
            found.setdefault(result.record["task_name"], []).append(result)
    chosen = {}
    for task, candidates in found.items():
        most = max(r.record["config"]["n_queries"] for r in candidates)
        best = [r for r in candidates if r.record["config"]["n_queries"] == most]
        if len(best) > 1:
            raise ValueError(
                f"{benchmark.name}: {task} has {len(best)} matching records: {[str(r.path) for r in best]}"
            )
        chosen[task] = best[0].record
    return chosen


def per_task_table(results: Results, benchmark: Benchmark, value: str = "rating") -> pd.DataFrame:
    """One row per model and one column per task, holding each record's `value` for the model: "rating", "ci_low",
    "ci_high" or "ndcg_at_10". A task without a record, or a model a record lacks, is missing."""
    import pandas as pd

    records = benchmark_records(results, benchmark)
    rows: dict[str, dict[str, float]] = {}
    for task, record in records.items():
        for rating in record["ratings"]:
            rows.setdefault(rating["model"], {})[task] = rating.get(value)
    table = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=list(benchmark.tasks))
    table.index.name = "Model"
    return table


def summary_table(results: Results, benchmark: Benchmark) -> pd.DataFrame:
    """One row per model: its rank, mean rating over the benchmark's tasks, and how many tasks it has. A model
    missing any task has no mean and no rank, as in mteb."""
    ratings = per_task_table(results, benchmark)
    tasks = ratings.notna().sum(axis=1)
    mean = ratings.mean(axis=1).where(tasks == len(benchmark.tasks))
    table = (
        mean.to_frame("Mean")
        .assign(Tasks=tasks)
        .sort_values(["Mean", "Tasks"], ascending=False, na_position="last")
        .reset_index()
    )
    table.insert(0, "Rank", table["Mean"].rank(ascending=False, method="min").astype("Int64"))
    return table


def agreement_table(results: Results, benchmark: Benchmark, bootstrap: int = 1000, seed: int = 0) -> pd.DataFrame:
    """For each task whose record was scored against the dataset's labels: Spearman's rho between the judge's
    ratings and the models' nDCG@10 on the same queries, with a bootstrap interval over models."""
    import pandas as pd

    from ..agreement import correlate

    rows = []
    for task, record in benchmark_records(results, benchmark).items():
        if record.get("labels") != "dataset":
            continue
        ratings = {r["model"]: r["rating"] for r in record["ratings"]}
        ndcg = {r["model"]: r["ndcg_at_10"] for r in record["ratings"] if r.get("ndcg_at_10") is not None}
        agreement = correlate(ratings, ndcg, bootstrap=bootstrap, seed=seed)
        low, high = agreement.get("spearman_ci95", [None, None])
        rows.append(
            {
                "Task": task,
                "Models": agreement.get("n_models"),
                "Spearman": agreement.get("spearman_rho"),
                "CI low": low,
                "CI high": high,
            }
        )
    order = {t: i for i, t in enumerate(benchmark.tasks)}
    return pd.DataFrame(rows, columns=["Task", "Models", "Spearman", "CI low", "CI high"]).sort_values(
        "Task", key=lambda s: s.map(order), ignore_index=True
    )
