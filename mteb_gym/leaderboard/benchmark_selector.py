"""The benchmark dropdown and the benchmark's description."""

from __future__ import annotations

import gradio as gr

from ..benchmarks import get_benchmarks
from ..benchmarks.benchmarks import GYM_ENG_V1

DEFAULT_BENCHMARK_NAME = GYM_ENG_V1.name


def _make_selector() -> tuple[gr.Dropdown, gr.Markdown]:
    selector = gr.Dropdown([b.name for b in get_benchmarks()], value=DEFAULT_BENCHMARK_NAME, label="Benchmark")
    return selector, gr.Markdown()
