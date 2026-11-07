"""The decision models that work by name. One small file per family: an object with `.model` and
`decide(kind, state, question, options=None)`, for the kinds "score" and "choice", which returns a probability per option. To add one, write its file and list
it in `known`. A new model in the format of JEV-9B and JEV-27B works without an entry, at its latest version."""

from __future__ import annotations

from ..local import repo_files
from . import decider, jev, systemone, v11
from .known import KNOWN

_FAMILIES = {"jev": jev, "decider": decider, "systemone": systemone, "v11": v11}
MODELS = {name: _FAMILIES[family].load for name, (family, _) in KNOWN.items()}


def decision_model_in_repo(name: str, files: set[str], url: str | None, api_key: str | None):
    """The decision model the files of a repo show, or None for any other model."""
    if files & {"decision_config.json", "decider_config.json"}:
        return systemone.load(name, url, api_key)
    return jev.load(name, url, api_key, files) if jev.is_jev_repo(files) else None


def DecisionModel(model: str, base_url: str, api_key: str | None = None):  # noqa: N802
    """The decision model served at `base_url`, as `LLM` is for a chat model. Use it for a model the library cannot tell is
    one from its name."""
    if model in MODELS:
        return MODELS[model](model, base_url, api_key)
    return decision_model_in_repo(model, repo_files(model) or set(), base_url, api_key) or systemone.load(
        model, base_url, api_key
    )
