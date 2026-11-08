"""What the library knows about a decision model it lists by name."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelMeta:
    """A model at the version that was tested.

    Attributes:
        name: The name on Hugging Face.
        revision: The commit that was tested. None where the model's server cannot be told one.
        loader: Returns the client: `loader(name, revision=..., url=..., api_key=..., reference=..., **loader_kwargs)`.
        loader_kwargs: Further arguments for the loader.
        reference: The model card, which says how to serve the model.
    """

    name: str
    revision: str | None
    loader: Callable
    loader_kwargs: dict[str, Any] = field(default_factory=dict)
    reference: str | None = None

    def load_model(self, url: str | None = None, api_key: str | None = None):
        return self.loader(
            self.name,
            revision=self.revision,
            url=url,
            api_key=api_key,
            reference=self.reference,
            **self.loader_kwargs,
        )
