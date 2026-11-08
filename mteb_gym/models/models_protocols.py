"""What a decision model offers the judges."""

from __future__ import annotations

from typing import Protocol


class DecisionClient(Protocol):
    """A decision model: a probability per option. `kind` is "score", for grades "0" to "5", or "choice", with
    `options`."""

    model: str

    def decide(self, kind: str, state: str, question: str, options: list[str] | None = None) -> dict[str, float]: ...
