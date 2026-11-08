"""The leaderboard, a Gradio app over a results repository, as mteb's leaderboard is over its results."""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from ..benchmarks import get_benchmark
from ..results import Results, load_results
from .benchmark_selector import _make_selector
from .table import (
    apply_agreement_styling_from_benchmark,
    apply_per_task_styling_from_benchmark,
    apply_summary_styling_from_benchmark,
)


def _load(source: str | None) -> Results:
    """The results repository by default; a local folder as it is; any other source as a repository's URL."""
    if source is None:
        return load_results()
    return load_results(source) if Path(source).exists() else load_results(results_repo=source)


def get_leaderboard_app(results: str | None = None) -> gr.Blocks:
    """The leaderboard over the results repository, or over `results`: a results folder or a repository's URL."""
    loaded = _load(results)

    def on_benchmark_select(name):
        benchmark = get_benchmark(name)
        agreement = apply_agreement_styling_from_benchmark(benchmark, loaded)
        return (
            benchmark.description or "",
            apply_summary_styling_from_benchmark(benchmark, loaded),
            apply_per_task_styling_from_benchmark(benchmark, loaded),
            gr.update(value=agreement, visible=agreement is not None),
        )

    with gr.Blocks(title="MTEB Gym Leaderboard") as app:
        gr.Markdown("# MTEB Gym Leaderboard")
        selector, description = _make_selector()
        with gr.Tab("Summary"):
            summary = gr.Dataframe(interactive=False)
        with gr.Tab("Per task"):
            per_task = gr.Dataframe(interactive=False)
        with gr.Tab("Agreement with labels"):
            agreement = gr.Dataframe(interactive=False, visible=False)
        outputs = [description, summary, per_task, agreement]
        selector.change(on_benchmark_select, selector, outputs)
        app.load(on_benchmark_select, selector, outputs)
    return app
