"""A decision model on a server with a `/v1/systemone` endpoint. The server builds the prompt and applies the model's
calibration, so nothing here depends on the model."""

from __future__ import annotations

import os

import httpx

from ..serving import post_json

_GRADES = [str(i) for i in range(6)]
_SERVE = {
    "LiquidAI/d1-3B": (
        'pip install "vllm>=0.32"\n'
        "vllm serve LiquidAI/d1-3B --logprobs-mode raw_logprobs --gpu-memory-utilization 0.6 --port 8000"
    ),
    "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610": (
        "pip install strands-decider\n"
        "strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --port 8000"
    ),
}


class SystemOne:
    how = "decision model"

    def __init__(self, model: str, url: str, api_key: str | None = None, timeout: float = 600.0, max_retries: int = 6):
        self.model = model
        self.base_url = url.rstrip("/").removesuffix("/v1")
        self.max_retries = max_retries
        key = api_key or os.environ.get("MTEB_GYM_API_KEY") or "EMPTY"
        self._http = httpx.Client(timeout=timeout, headers={"Authorization": f"Bearer {key}"})

    def decide(self, kind: str, state: str, question: str, options: list[str] | None = None) -> dict[str, float]:
        criteria = {"score": _GRADES, "choice": dict.fromkeys(options or ()), "noul": None}[kind]
        spec = {"type": kind, "instructions": question, **({"criteria": criteria} if criteria else {})}
        body = {"model": self.model, "state": state, "questions": {"q": spec}}
        try:
            answer = post_json(self._http, f"{self.base_url}/v1/systemone", body, self.max_retries)["answers"]["q"]
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (404, 501):
                raise ValueError(f"{self.base_url} does not answer decisions for {self.model} on /v1/systemone.") from e
            raise
        if kind == "noul":
            return {"false": 1 - answer["noul"], "true": answer["noul"]}
        return {o: answer["probabilities"][o] for o in (options or _GRADES)}


def load(name: str, url: str | None, api_key: str | None):
    if not url:
        how = f":\n{_SERVE[name]}" if name in _SERVE else " as its model card describes."
        raise ValueError(f"{name} cannot be loaded directly. Pass judge_url after serving it{how}")
    return SystemOne(name, url, api_key=api_key)
