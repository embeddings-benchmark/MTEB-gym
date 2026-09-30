"""LLM clients: LLM for any OpenAI-compatible /chat/completions endpoint (vLLM, Ollama,
OpenAI, Together, OpenRouter, and the compatible endpoints of Anthropic and Gemini),
MockLLM for tests and dry runs. A client implements chat(messages) -> str. Answers are asked for
as JSON in the prompt and read from the text, so a model may reason before it answers. Sampling
is the model's own: its authors' defaults, unless the client was built with a temperature."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re

logger = logging.getLogger(__name__)


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
        if "system a" in prompt.lower():
            return json.dumps({"winner": ["A", "B", "tie"][h % 3], "confidence": "low", "reasoning": "mock"})
        words = re.findall(r"[A-Za-z]{4,}", prompt.split("[1] ", 1)[-1])[:6]  # from the first shown document
        return json.dumps({"query": "what is known about " + (" ".join(words) or f"topic {h % 1000}")})


class LLM:
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
        from openai import OpenAI

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if base_url is None and key is None and not os.environ.get("OPENAI_BASE_URL"):
            raise ValueError(
                f"gym.LLM({model!r}): no API key and no base_url. Set OPENAI_API_KEY for OpenAI, pass base_url "
                f"and api_key for another provider, or serve the model yourself (`vllm serve {model}`) and pass "
                "base_url='http://localhost:8000/v1'."
            )
        # The SDK retries connection errors, timeouts, 429 and 5xx with backoff. Anything else
        # (bad key, unknown model) raises at once. timeout: a hung call would otherwise stall a worker forever.
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
                return resp.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001
                # Some models refuse a parameter and answer 400 naming it, e.g. a temperature on a model
                # that always reasons. Drop it, remember, and run at its defaults.
                rejected = [k for k in params if k in str(e) and k not in self._rejected]
                if getattr(e, "status_code", None) != 400 or not rejected:
                    raise
                self._rejected.update(rejected)
                logger.warning("%s refuses %s; running it at the model's own defaults", self.model, ", ".join(rejected))


def llm_settings(client) -> dict:
    """How an LLM client is configured: the model, the endpoint, and the settings it sends, which are
    the settings its answers' cache identity is built from, so a live run and a replay of cached
    answers record the same. Settings a server refused, and the model it reported, are known only
    from live calls."""
    base_url = getattr(getattr(client, "client", None), "base_url", None)
    configured = {
        "temperature": getattr(client, "temperature", None),
        "max_completion_tokens": getattr(client, "max_tokens", None),
    }
    settings = {
        "model": getattr(client, "model", str(client)),
        "base_url": str(base_url) if base_url else None,
        "served_model": getattr(client, "served_model", None),
        **{k: v for k, v in configured.items() if v is not None},
        **(getattr(client, "extra_body", None) or {}),
    }
    refused = sorted(getattr(client, "_rejected", ()))
    return {**settings, "refused": refused} if refused else settings
