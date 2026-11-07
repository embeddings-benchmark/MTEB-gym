"""A decision model on a server with a `/v1/systemone` endpoint. The server builds the prompt and applies the model's
calibration, so nothing here depends on the model."""

from __future__ import annotations

import httpx

from ..serving import GRADES, post_json, server_client, server_url
from .model_meta import ModelMeta


class SystemOne:
    how = "decision model"

    def __init__(self, model: str, url: str, api_key: str | None = None, max_retries: int = 6):
        self.model = model
        self.base_url = server_url(url)
        self.max_retries = max_retries
        self._http = server_client(api_key)

    def decide(self, kind: str, state: str, question: str, options: list[str] | None = None) -> dict[str, float]:
        criteria = {"score": GRADES, "choice": dict.fromkeys(options or ())}[kind]
        spec = {"type": kind, "instructions": question, "criteria": criteria}
        body = {"model": self.model, "state": state, "questions": {"q": spec}}
        try:
            answer = post_json(self._http, f"{self.base_url}/v1/systemone", body, self.max_retries)["answers"]["q"]
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (404, 501):
                raise ValueError(f"{self.base_url} does not answer decisions for {self.model} on /v1/systemone.") from e
            raise
        return {o: answer["probabilities"][o] for o in (options or GRADES)}


def load(name: str, url: str | None, api_key: str | None, revision: str | None = None):
    """A server says which version it runs, so `revision` only goes in the command."""
    if not url:
        how = f":\n{_SERVE[name]}" if name in _SERVE else " as its model card describes."
        raise ValueError(f"{name} cannot be loaded directly. Pass its URL after serving it{how}")
    return SystemOne(name, url, api_key=api_key)


d1_3b = ModelMeta(loader=load, name="LiquidAI/d1-3B", revision="051bcc464b01b9f92942b364d9586b0ef5912432")
# its server cannot take a revision
strands_decider_2b = ModelMeta(loader=load, name="StrandsAgents/strands-decider-2B-qwen3.5-v1-2610", revision=None)

_SERVE = {
    d1_3b.name: (
        'pip install "vllm>=0.32"\n'
        f"vllm serve {d1_3b.name} --revision {d1_3b.revision} --logprobs-mode raw_logprobs "
        "--gpu-memory-utilization 0.6 --port 8000"
    ),
    strands_decider_2b.name: f"pip install strands-decider\nstrands-decider serve {strands_decider_2b.name} --port 8000",
}
