"""Clients: LLM for any OpenAI-compatible /chat/completions endpoint, DecisionModel for a decision model on a
server, MockLLM for tests and dry runs. Answers are requested as JSON in the prompt and read from the text, so a model
may reason before it answers. Sampling is the model's own unless the client is built with a temperature.

An explicit base_url never receives OPENAI_API_KEY: pass api_key, or set MTEB_GYM_API_KEY."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from typing import Protocol

import httpx
from openai import OpenAI

from .local import LocalChatModel, repo_files, shared
from .models import MODELS, decision_model_in_repo
from .serving import InputTooLong, _too_long

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
    how = "mock model"

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

    how = "chat model"

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
                if _too_long(getattr(e, "status_code", None), str(e)):
                    raise InputTooLong(str(e)) from e
                # A model may refuse a parameter with a 400 naming it: drop it, and run at its defaults.
                rejected = [k for k in params if k in str(e) and k not in self._rejected]
                if getattr(e, "status_code", None) != 400 or not rejected:
                    raise
                self._rejected.update(rejected)
                logger.warning("%s refuses %s; running it at the model's own defaults", self.model, ", ".join(rejected))


def _serves_a_decision_module(url: str, api_key: str | None) -> bool:
    key = api_key or os.environ.get("MTEB_GYM_API_KEY") or "EMPTY"
    models = f"{url.rstrip('/').removesuffix('/v1')}/v1/models"
    try:
        served = httpx.get(models, headers={"Authorization": f"Bearer {key}"}, timeout=30).json()["data"]
        return "jev-decision" in {m["id"] for m in served}
    except (httpx.HTTPError, ValueError, KeyError, TypeError):  # any other kind of server: not one
        return False


def load_model(model, url: str | None = None, api_key: str | None = None, extra_body: dict | None = None):
    """A generator or judge by name. A decision model is known by name (see `models`) or by the files of its Hugging Face
    repo. Any other name with `url` is the chat model served there; without it, a model on Hugging Face is loaded here on
    first use, and any other goes to the API. Anything that is not a name is returned as it is."""
    if not isinstance(model, str):
        return model
    if model == "mock":
        return MockLLM()
    if model in MODELS:
        return MODELS[model](model, url, api_key)
    if url:
        files = repo_files(model)
        if files is None and _serves_a_decision_module(url, api_key):
            raise ValueError(
                f"{url} serves a decision model, but {model!r} cannot be read as a Hugging Face repo. "
                "Use the model's repo name, such as autotrust/JEV-9B. If offline, it must be in the local cache."
            )
        decision = decision_model_in_repo(model, files or set(), url, api_key)
        return decision or LLM(model, base_url=url, api_key=api_key, extra_body=extra_body)
    if api_key or os.environ.get("OPENAI_BASE_URL"):
        return LLM(model, api_key=api_key, extra_body=extra_body)
    files = repo_files(model)
    if files is not None:
        return decision_model_in_repo(model, files, None, None) or shared(LocalChatModel, model)
    if os.environ.get("OPENAI_API_KEY"):
        return LLM(model, extra_body=extra_body)
    raise ValueError(
        f"{model!r} was not found on Hugging Face or in its local cache. Use a model that is there, set OPENAI_API_KEY to reach it through an API, "
        "or pass the URL where it is served."
    )


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
