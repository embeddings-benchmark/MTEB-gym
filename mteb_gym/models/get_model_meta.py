"""The decision models the library lists, collected from every module in `model_implementations`."""

from __future__ import annotations

import importlib
import pkgutil

from . import model_implementations
from .model_meta import ModelMeta


def _all_model_metas() -> list[ModelMeta]:
    metas = []
    for module_info in pkgutil.iter_modules(model_implementations.__path__):
        module = importlib.import_module(f"{model_implementations.__name__}.{module_info.name}")
        metas += [value for value in vars(module).values() if isinstance(value, ModelMeta)]
    return metas


MODEL_REGISTRY: dict[str, ModelMeta] = {meta.name: meta for meta in _all_model_metas()}


def get_model_metas() -> list[ModelMeta]:
    return list(MODEL_REGISTRY.values())


def get_model_meta(name: str) -> ModelMeta:
    return MODEL_REGISTRY[name]
