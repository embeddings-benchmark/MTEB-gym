from __future__ import annotations

from ..model_meta import ModelMeta
from ..systemone_wrapper import SystemOne

d1_3b = ModelMeta(
    loader=SystemOne,
    name="LiquidAI/d1-3B",
    revision="051bcc464b01b9f92942b364d9586b0ef5912432",
    reference="https://huggingface.co/LiquidAI/d1-3B",
)

strands_decider_2b = ModelMeta(
    loader=SystemOne,
    name="StrandsAgents/strands-decider-2B-qwen3.5-v1-2610",
    revision=None,  # its server cannot take one
    reference="https://huggingface.co/StrandsAgents/strands-decider-2B-qwen3.5-v1-2610",
)
