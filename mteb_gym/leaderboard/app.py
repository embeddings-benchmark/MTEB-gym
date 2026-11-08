"""The leaderboard, a Gradio app over a results repository, as mteb's leaderboard is over its results."""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import pandas as pd

from ..benchmarks import agreement_table, get_benchmark, get_benchmarks, per_task_table, summary_table
from ..results import Results, load_results

DEFAULT_BENCHMARK = "Gym(eng, v1)"


def _results(source: str | None) -> Results:
    """The results repository by default; a local folder as it is; any other source as a repository's URL."""
    if source is None:
        return load_results()
    return load_results(source) if Path(source).exists() else load_results(results_repo=source)


def _tables(results: Results, name: str) -> tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    benchmark = get_benchmark(name)
    summary = summary_table(results, benchmark).round({"Mean": 1})
    per_task = per_task_table(results, benchmark).round(1).reset_index()
    agreement = agreement_table(results, benchmark).round(3) if benchmark.query_source == "original" else None
    return benchmark.description or "", summary, per_task, agreement


def get_leaderboard_app(results: str | None = None) -> gr.Blocks:
    """The leaderboard over the results repository, or over `results`: a results folder or a repository's URL."""
    loaded = _results(results)
    names = [b.name for b in get_benchmarks()]

    def show(name):
        description, summary, per_task, agreement = _tables(loaded, name)
        return description, summary, per_task, gr.update(value=agreement, visible=agreement is not None)

    with gr.Blocks(title="MTEB Gym Leaderboard") as app:
        gr.Markdown("# MTEB Gym Leaderboard")
        benchmark = gr.Dropdown(names, value=DEFAULT_BENCHMARK, label="Benchmark")
        description = gr.Markdown()
        with gr.Tab("Summary"):
            summary = gr.Dataframe(interactive=False)
        with gr.Tab("Per task"):
            per_task = gr.Dataframe(interactive=False)
        with gr.Tab("Agreement with labels"):
            agreement = gr.Dataframe(interactive=False, visible=False)
        outputs = [description, summary, per_task, agreement]
        benchmark.change(show, benchmark, outputs)
        app.load(show, benchmark, outputs)
    return app
