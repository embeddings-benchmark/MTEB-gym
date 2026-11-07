"""What the library knows about a decision model it lists by name."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelMeta:
    """A model at the version that was tested. `loader(name, url, api_key, revision=...)` returns the client. The revision
    is None where a model's server cannot be told one."""

    name: str
    revision: str | None
    loader: Callable

    def load_model(self, url: str | None = None, api_key: str | None = None):
        return self.loader(self.name, url, api_key, revision=self.revision)
