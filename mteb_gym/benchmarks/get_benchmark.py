from __future__ import annotations

import difflib
from functools import cache

from .benchmark import Benchmark


@cache
def _registry() -> dict[str, Benchmark]:
    from . import benchmarks

    return {value.name: value for value in vars(benchmarks).values() if isinstance(value, Benchmark)}


def get_benchmark(name: str) -> Benchmark:
    registry = _registry()
    if name not in registry:
        close = difflib.get_close_matches(name, registry)
        raise KeyError(f"'{name}' not found." + (f" Did you mean: {close[0]}?" if close else ""))
    return registry[name]


def get_benchmarks(names: list[str] | None = None) -> list[Benchmark]:
    return [get_benchmark(name) for name in names] if names is not None else list(_registry().values())
