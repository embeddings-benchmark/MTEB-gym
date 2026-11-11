"""Decision models: a wrapper for each way of reaching one, and in `model_implementations` a `ModelMeta` for each model
the library knows by name."""

from __future__ import annotations

from ..local import repo_files
from .get_model_meta import MODEL_REGISTRY, get_model_meta, get_model_metas
from .jev_wrapper import is_jev_repo, jev_loader
from .model_meta import ModelMeta
from .models_protocols import DecisionClient
from .systemone_wrapper import SystemOne

__all__ = [
    "MODEL_REGISTRY",
    "DecisionModel",
    "ModelMeta",
    "decision_model_in_repo",
    "get_model_meta",
    "get_model_metas",
]


def decision_model_in_repo(name: str, files: set[str], url: str | None, api_key: str | None) -> DecisionClient | None:
    """The decision model the files of a repo show, or None for any other model."""
    if files & {"decision_config.json", "decider_config.json"}:
        return SystemOne(name, url=url, api_key=api_key)
    return jev_loader(name, url=url, api_key=api_key, files=files) if is_jev_repo(files) else None


def DecisionModel(model: str, base_url: str, api_key: str | None = None) -> DecisionClient:  # noqa: N802
    """The decision model served at `base_url`, as `LLM` is for a chat model. Use it for a model the library cannot tell is
    one from its name."""
    if model in MODEL_REGISTRY:
        return MODEL_REGISTRY[model].load_model(base_url, api_key)
    files = repo_files(model) or set()
    return decision_model_in_repo(model, files, base_url, api_key) or SystemOne(model, url=base_url, api_key=api_key)
