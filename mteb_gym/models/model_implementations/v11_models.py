from __future__ import annotations

from ..model_meta import ModelMeta
from ..v11_wrapper import V11Model

jev_27b_v1_1 = ModelMeta(
    loader=V11Model,
    loader_kwargs=dict(served_name="AutoTrust-JEV-27B-v1.1"),  # the name its card's script serves it under
    name="autotrust/AutoTrust-JEV-27B-v1.1",
    revision="ae156900e9f338e423042b89e25d0fd8093064d4",
    reference="https://huggingface.co/autotrust/AutoTrust-JEV-27B-v1.1",
)
