"""The decision models that work by name. One small file per family: an object with `.model` and
`decide(kind, state, question, options=None)`, which returns a probability per option. To add one, write its file and list
it here. A new model in the format of JEV-9B and JEV-27B needs no entry."""

from __future__ import annotations

from . import decider, jev, v11

MODELS = {
    "Mapika/decider-4b": decider.load,
    "autotrust/AutoTrust-JEV-27B-v1.1": v11.load,
}


def decision_model_in_repo(name: str, files: set[str], url: str | None, api_key: str | None):
    """The decision model the files of a repo show, or None for any other model. A format not read here is refused, so a
    decision model never ends up as a chat model."""
    if files & {"decision_config.json", "decider_config.json"}:
        raise ValueError(
            f"{name} is a decision model in a format not supported yet. Add it in mteb_gym/models, or pass an object "
            "with decide() as the judge."
        )
    return jev.load(name, url, api_key, files) if jev.is_jev_repo(files) else None
