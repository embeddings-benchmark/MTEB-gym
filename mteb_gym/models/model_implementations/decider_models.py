from __future__ import annotations

from ..decider_wrapper import decider_loader
from ..model_meta import ModelMeta

decider_4b = ModelMeta(
    loader=decider_loader,
    extra_requirements_groups=["decider"],
    name="Mapika/decider-4b",
    revision="585473dec4fa93f0325c2bf02dda7273b61f44c1",
    reference="https://huggingface.co/Mapika/decider-4b",
)
