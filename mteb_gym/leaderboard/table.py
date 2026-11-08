"""A benchmark's tables as the leaderboard shows them."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from ..benchmarks import Benchmark
    from ..results import Results


def apply_summary_styling_from_benchmark(benchmark: Benchmark, results: Results) -> pd.DataFrame:
    return results.get_benchmark_result(benchmark).round({"Mean": 1})


def apply_per_task_styling_from_benchmark(benchmark: Benchmark, results: Results) -> pd.DataFrame:
    return benchmark._create_per_task_table(results).round(1).reset_index()


def apply_agreement_styling_from_benchmark(benchmark: Benchmark, results: Results) -> pd.DataFrame | None:
    """Agreement with labels, for benchmarks on the tasks' own queries."""
    if benchmark.query_source != "original":
        return None
    return benchmark._create_agreement_table(results).round(3)
