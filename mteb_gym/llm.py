"""Clients: LLM for any OpenAI-compatible /chat/completions endpoint, DecisionModel for a decision model's /v1/decide
endpoint, MockLLM for tests and dry runs. Answers are requested as JSON in the prompt and read from the text, so a model
may reason before it answers. Sampling is the model's own unless the client is built with a temperature.

An explicit base_url never receives OPENAI_API_KEY: pass api_key, or set MTEB_GYM_API_KEY."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from typing import Protocol

import httpx
from openai import OpenAI

logger = logging.getLogger(__name__)


class ChatClient(Protocol):
    """What a generator or an LLM judge needs."""

    def chat(self, messages: list[dict]) -> str: ...


class DecisionClient(Protocol):
    """What a decision-model judge needs."""

    def decide(self, kind: str, state: str, question: str, options: list[str] | None = None) -> dict[str, float]: ...


class MockLLM:
    """Deterministic stand-in for tests and dry runs: same input, same output."""

    model = "mock"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def _hash(self, text: str) -> int:
        return int(hashlib.sha256(f"{self.seed}:{text}".encode()).hexdigest()[:8], 16)

    def chat(self, messages: list[dict]) -> str:
        prompt = " ".join(m.get("content", "") for m in messages)
        h = self._hash(prompt)
        if "rate the quality" in prompt.lower():
            return json.dumps({"score": (h % 5) + 1, "reason": "mock"})
        if "grade how relevant" in prompt.lower():
            return json.dumps({"reasoning": "mock", "grade": h % 6})
        if "system a" in prompt.lower():
            return json.dumps({"winner": ["A", "B", "tie"][h % 3], "confidence": "low", "reasoning": "mock"})
        words = re.findall(r"[A-Za-z]{4,}", prompt.split("[1] ", 1)[-1])[:6]  # from the first shown document
        return json.dumps({"query": "what is known about " + (" ".join(words) or f"topic {h % 1000}")})


class LLM:
    """A chat model behind an OpenAI-compatible endpoint."""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        max_retries: int = 4,
        extra_body: dict | None = None,
        timeout: float = 120.0,
    ):
        key = api_key or (os.environ.get("MTEB_GYM_API_KEY") if base_url else os.environ.get("OPENAI_API_KEY"))
        if base_url is None and key is None and not os.environ.get("OPENAI_BASE_URL"):
            raise ValueError(
                f"LLM({model!r}): no API key and no base_url. Set OPENAI_API_KEY for OpenAI, pass base_url "
                f"and api_key for another provider, or serve the model yourself (`vllm serve {model}`) and pass "
                "base_url='http://localhost:8000/v1'."
            )
        # The SDK retries connection errors, timeouts, 429 and 5xx with backoff; anything else raises at once.
        # "EMPTY" is the conventional key for local servers, which accept anything.
        self.client = OpenAI(base_url=base_url, api_key=key or "EMPTY", timeout=timeout, max_retries=max_retries)
        self.model = model
        self.temperature = temperature  # unset: the model's own default, e.g. its generation_config.json
        self.max_tokens = max_tokens  # no cap unless asked: a cap also counts a reasoning model's thinking
        self.extra_body = extra_body  # server knobs, e.g. {"chat_template_kwargs": {"enable_thinking": False}}
        self._rejected: set[str] = set()  # sampling parameters this model refused
        self.served_model: str | None = None  # the model string the server reported, e.g. a dated snapshot

    def chat(self, messages: list[dict]) -> str:
        params = {}
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.max_tokens is not None:
            params["max_completion_tokens"] = self.max_tokens
        while True:
            try:
                sent = {k: v for k, v in params.items() if k not in self._rejected}
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages, extra_body=self.extra_body, **sent
                )
                self.served_model = getattr(resp, "model", None) or self.served_model
                choice = resp.choices[0]
                if getattr(choice, "finish_reason", None) == "length":  # cut off by a token limit: no usable answer
                    logger.warning("%s hit its token limit; the answer is left out", self.model)
                    return ""
                return choice.message.content or ""
            except Exception as e:
                # A model may refuse a parameter with a 400 naming it: drop it, and run at its defaults.
                rejected = [k for k in params if k in str(e) and k not in self._rejected]
                if getattr(e, "status_code", None) != 400 or not rejected:
                    raise
                self._rejected.update(rejected)
                logger.warning("%s refuses %s; running it at the model's own defaults", self.model, ", ".join(rejected))


class DecisionModel:
    """A decision model behind a `/v1/decide` endpoint. `decide("score", state, question)` returns a probability for
    each grade "0" to "5", in one forward pass."""

    def __init__(
        self, model: str, base_url: str, api_key: str | None = None, timeout: float = 600.0, max_retries: int = 6
    ):
        self.model = model
        self.base_url = base_url.rstrip("/").removesuffix("/v1")
        self.max_retries = max_retries
        key = api_key or os.environ.get("MTEB_GYM_API_KEY") or "EMPTY"
        self._http = httpx.Client(timeout=timeout, headers={"Authorization": f"Bearer {key}"})

    def decide(self, kind: str, state: str, question: str, options: list[str] | None = None) -> dict[str, float]:
        body = {"kind": kind, "state": state, "question": question, **({"options": options} if options else {})}
        # a busy server queues requests, so the timeout is long and retries back off; a client error is final
        error = ""
        for attempt in range(self.max_retries + 1):
            if attempt:
                time.sleep(min(60, 2 ** (attempt - 1)))
            try:
                r = self._http.post(f"{self.base_url}/v1/decide", json=body)
            except httpx.TransportError as e:
                error = repr(e)
                continue
            if r.status_code >= 500 or r.status_code in (408, 429):
                error = f"HTTP {r.status_code}: {r.text[:200]}"
                continue
            r.raise_for_status()
            try:
                out = r.json()
            except ValueError:
                error = f"not JSON: {r.text[:200]}"
                continue
            return dict(zip(out["options"], out["probabilities"], strict=True))
        raise RuntimeError(f"{self.base_url}/v1/decide failed after {self.max_retries + 1} attempts: {error}")


def llm_settings(client) -> dict:
    """How a client is configured, for the record. The endpoint is left out: it can be private."""
    configured = {
        "temperature": getattr(client, "temperature", None),
        "max_completion_tokens": getattr(client, "max_tokens", None),
    }
    settings = {
        "model": getattr(client, "model", str(client)),
        "served_model": getattr(client, "served_model", None),
        **{k: v for k, v in configured.items() if v is not None},
        **(getattr(client, "extra_body", None) or {}),
    }
    refused = sorted(getattr(client, "_rejected", ()))
    return {**settings, "refused": refused} if refused else settings
