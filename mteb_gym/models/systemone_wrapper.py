"""A decision model on a server with a `/v1/systemone` endpoint. The server builds the prompt and applies the model's
calibration, so nothing here depends on the model. A server says which version it runs, so the revision is not used."""

from __future__ import annotations

import httpx

from ..serving import GRADES, model_card, post_json, server_client, server_url


class SystemOne:
    how = "decision model"

    def __init__(
        self,
        name: str,
        revision: str | None = None,
        url: str | None = None,
        api_key: str | None = None,
        reference: str | None = None,
        max_retries: int = 6,
    ):
        if not url:
            raise ValueError(
                f"{name} cannot be loaded directly. Serve it as its model card describes, then pass its URL: "
                f"{model_card(name, reference)}"
            )
        self.model = name
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
