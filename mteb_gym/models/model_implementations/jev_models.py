from __future__ import annotations

from ..jev_wrapper import jev_loader
from ..model_meta import ModelMeta

jev_9b = ModelMeta(
    loader=jev_loader,
    extra_requirements_groups=["transformers5"],  # its architecture is newer than transformers 4
    name="autotrust/JEV-9B",
    revision="7ac669b8fb476ba926491e3d1c39f9330d68dbed",
    reference="https://huggingface.co/autotrust/JEV-9B",
)

jev_27b = ModelMeta(
    loader=jev_loader,
    extra_requirements_groups=["transformers5"],  # its architecture is newer than transformers 4
    name="autotrust/JEV-27B",
    revision="14b425269ed24452e0a5ad5a8f3e51d139647a88",
    reference="https://huggingface.co/autotrust/JEV-27B",
)

jev_35b = ModelMeta(
    loader=jev_loader,
    extra_requirements_groups=["transformers5"],  # its architecture is newer than transformers 4
    name="autotrust/JEV-35B",
    revision="a5d0d580bb9324f0f4c07dcba5c5ca91e813e7c9",
    reference="https://huggingface.co/autotrust/JEV-35B",
)

gev_26b = ModelMeta(
    loader=jev_loader,
    extra_requirements_groups=["transformers5"],  # its architecture is newer than transformers 4
    name="autotrust/GEV-26B-Decide",
    revision="eea849038e792cd09aa731d59d586f0eaf2c18f6",
    reference="https://huggingface.co/autotrust/GEV-26B-Decide",
)
