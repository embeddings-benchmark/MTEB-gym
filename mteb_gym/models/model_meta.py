"""What the library knows about a decision model it lists by name."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, requires, version
from typing import Any

from packaging.requirements import Requirement
from packaging.version import InvalidVersion, Version

from .models_protocols import DecisionClient

INSTALL = "git+https://github.com/embeddings-benchmark/MTEB-gym.git"


@dataclass(frozen=True)
class ModelMeta:
    """A model at the version that was tested.

    Attributes:
        name: The name on Hugging Face.
        revision: The commit that was tested. None where the model's server cannot be told one.
        loader: Returns the client: `loader(name, revision=..., url=..., api_key=..., reference=..., **loader_kwargs)`.
        loader_kwargs: Further arguments for the loader.
        extra_requirements_groups: The extras of mteb-gym it needs to run in this process.
        reference: The model card, which says how to serve the model.
    """

    name: str
    revision: str | None
    loader: Callable[..., DecisionClient]
    loader_kwargs: dict[str, Any] = field(default_factory=dict)
    extra_requirements_groups: list[str] = field(default_factory=list)
    reference: str | None = None

    def load_model(self, url: str | None = None, api_key: str | None = None) -> DecisionClient:
        if url is None:  # the extras are for running it here: a server brings its own
            self._check_requirements()
        return self.loader(
            self.name,
            revision=self.revision,
            url=url,
            api_key=api_key,
            reference=self.reference,
            **self.loader_kwargs,
        )

    def _check_requirements(self) -> None:
        missing = self._missing_dependencies(self.extra_requirements_groups)
        if missing:
            groups = ",".join(self.extra_requirements_groups)
            raise ImportError(
                f"Model {self.name} is missing required dependencies: {', '.join(missing)}.\n"
                f'You can install it with `pip install "mteb-gym[{groups}] @ {INSTALL}"`.'
            )

    @staticmethod
    def _missing_dependencies(groups: list[str]) -> list[str]:
        """The requirements of the given extras that the installed packages do not satisfy, as mteb checks its own."""
        if not groups:
            return []
        try:
            specs = requires("mteb-gym") or []
        except PackageNotFoundError:  # run from a source folder: nothing to check against
            return []
        missing = []
        for spec in specs:
            req = Requirement(spec)
            if req.marker is None or not any(req.marker.evaluate({"extra": g}) for g in groups):
                continue
            try:
                installed = version(req.name)
            except PackageNotFoundError:
                missing.append(spec)
                continue
            try:
                if req.specifier and Version(installed) not in req.specifier:
                    missing.append(f"{spec} (installed: {installed})")
            except InvalidVersion:
                missing.append(f"{spec} (installed: {installed})")
        return missing
