"""Shared interfaces for model clients and cached model constructors."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, TypeVar

_ModelT = TypeVar("_ModelT")


class ChatClient(Protocol):
    """What a generator or an LLM judge needs."""

    def chat(self, messages: list[dict[str, Any]]) -> str: ...


class DecisionClient(Protocol):
    """A decision model: a probability per option. `kind` is "score", for grades "0" to "5", or "choice", with
    `options`."""

    model: str

    def decide(self, kind: str, state: str, question: str, options: list[str] | None = None) -> dict[str, float]: ...


Client = ChatClient | DecisionClient


class _SharedModels(Protocol):
    def __call__(self, cls: Callable[..., _ModelT], name: str, *args: Any) -> _ModelT: ...

    def cache_clear(self) -> None: ...
