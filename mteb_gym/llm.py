"""Clients: LLM for any OpenAI-compatible /chat/completions endpoint, DecisionModel for a decision model on a
server, MockLLM for tests and dry runs. Answers are requested as JSON in the prompt and read from the text, so a model
may reason before it answers. Sampling is the model's own unless the client is built with a temperature.

An explicit base_url never receives OPENAI_API_KEY: pass api_key, or set MTEB_GYM_API_KEY."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import threading
import time
from typing import Protocol

import httpx
from huggingface_hub import hf_hub_download
from openai import OpenAI

from .local import decision_prompt, is_decision_repo, load_local, refuse_other_decision_formats, repo_files

logger = logging.getLogger(__name__)

_READ_ONE_TOKEN = {  # the full distribution of the next token: a model's generation defaults must not truncate it
    "max_tokens": 1,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "min_p": 0.0,
    "repetition_penalty": 1.0,
    "add_special_tokens": False,
    "return_tokens_as_token_ids": True,
}
_MODULE_FLAGS = "--enable-lora --max-lora-rank 32 --lora-modules jev-decision=<model folder>/adapter_vllm"
_LOGPROBS_FLAG = "--logprobs-mode processed_logprobs"


class InputTooLong(Exception):
    """The input does not fit the model's context."""


def _too_long(status: int | None, message: str) -> bool:
    message = message.lower()
    return status == 400 and ("maximum context length" in message or "context_length_exceeded" in message)


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


class DecisionModel:
    """A decision model on a server, by its model card's recipe: the server returns the log-probability of each option's
    token, and the head's bias and the temperature are applied here. `decide("score", state, question)` returns a
    probability for each grade "0" to "5", and `decide("choice", state, question, options)` one for each option, in one
    forward pass. Nothing is fetched until the first decision, so judgments already cached need no server."""

    how = "decision model"

    def __init__(
        self, model: str, base_url: str, api_key: str | None = None, timeout: float = 600.0, max_retries: int = 6
    ):
        self.model = model
        self.base_url = base_url.rstrip("/").removesuffix("/v1")
        self.max_retries = max_retries
        key = api_key or os.environ.get("MTEB_GYM_API_KEY") or "EMPTY"
        self._http = httpx.Client(timeout=timeout, headers={"Authorization": f"Bearer {key}"})
        self._lock = threading.Lock()
        self._head = None

    def _prepare(self) -> None:
        """The two small files of the repo, and whether the server has what they describe."""
        with self._lock:
            if self._head is not None:
                return
            with open(hf_hub_download(self.model, "adapter_vllm/decision_head.json")) as f:
                head = json.load(f)
            with open(hf_hub_download(self.model, "calibration.json")) as f:
                self._temperature = json.load(f)["per_kind"]
            merged = head.get("merged")  # the decision model is in the weights: no separate module on the server
            self._served_as = self.model if merged else "jev-decision"
            flags = f"--served-model-name {self.model}" if merged else _MODULE_FLAGS
            r = self._http.get(f"{self.base_url}/v1/models")
            r.raise_for_status()
            if self._served_as not in {m["id"] for m in r.json()["data"]}:
                raise ValueError(
                    f"{self.base_url} does not serve {self._served_as}. Restart it with {flags} {_LOGPROBS_FLAG}"
                )
            self._head = head

    def decide(self, kind: str, state: str, question: str, options: list[str] | None = None) -> dict[str, float]:
        self._prepare()
        options, text = decision_prompt(kind, state, question, options)
        start = self._head["slots"]["ranges"][kind][0]
        ids = self._head["verbalizer_ids"][start : start + len(options)]
        out = self._post(
            "/v1/completions",
            {
                "model": self._served_as,
                "prompt": ("<bos>" if self._head.get("prefix_token") is not None else "") + text,
                "logprobs": len(options),
                "allowed_token_ids": ids,
                **_READ_ONE_TOKEN,
            },
        )
        found = {int(k.split(":")[1]): v for k, v in out["choices"][0]["logprobs"]["top_logprobs"][0].items()}
        if not set(ids) <= set(found):
            raise ValueError(f"{self.base_url} left out options' probabilities. Restart it with {_LOGPROBS_FLAG}")
        bias = self._head["bias"][start : start + len(options)]
        logits = [(found[i] + b) / self._temperature[kind] for i, b in zip(ids, bias, strict=True)]
        largest = max(logits)
        weights = [math.exp(z - largest) for z in logits]
        return {o: w / sum(weights) for o, w in zip(options, weights, strict=True)}

    def _post(self, path: str, body: dict) -> dict:
        # a busy server queues requests, so the timeout is long and retries back off; a client error is final
        error = ""
        for attempt in range(self.max_retries + 1):
            if attempt:
                time.sleep(min(60, 2 ** (attempt - 1)))
            try:
                r = self._http.post(f"{self.base_url}{path}", json=body)
            except httpx.TransportError as e:
                error = repr(e)
                continue
            if r.status_code >= 500 or r.status_code in (408, 429):
                error = f"HTTP {r.status_code}: {r.text[:200]}"
                continue
            if _too_long(r.status_code, r.text):
                raise InputTooLong(r.text[:200])
            r.raise_for_status()
            try:
                return r.json()
            except ValueError:
                error = f"not JSON: {r.text[:200]}"
        raise RuntimeError(f"{self.base_url}{path} failed after {self.max_retries + 1} attempts: {error}")


def _serves_a_decision_module(url: str, api_key: str | None) -> bool:
    key = api_key or os.environ.get("MTEB_GYM_API_KEY") or "EMPTY"
    models = f"{url.rstrip('/').removesuffix('/v1')}/v1/models"
    try:
        served = httpx.get(models, headers={"Authorization": f"Bearer {key}"}, timeout=30).json()["data"]
        return "jev-decision" in {m["id"] for m in served}
    except (httpx.HTTPError, ValueError, KeyError, TypeError):  # any other kind of server: not one
        return False


def load_model(model, url: str | None = None, api_key: str | None = None, extra_body: dict | None = None):
    """A generator or judge by name. With `url`, the model served there: a decision model if the name's Hugging Face repo
    is one, otherwise a chat model. Without `url`, a model on Hugging Face is
    loaded here on first use, and any other goes to the API. Anything that is not a name is returned as it is."""
    if not isinstance(model, str):
        return model
    if model == "mock":
        return MockLLM()
    if url:
        files = repo_files(model)
        if files is None:
            if _serves_a_decision_module(url, api_key):
                raise ValueError(
                    f"{url} serves a decision model, but {model!r} cannot be read as a Hugging Face repo. "
                    "Use the model's repo name, such as autotrust/JEV-9B. If offline, it must be in the local cache."
                )
            return LLM(model, base_url=url, api_key=api_key, extra_body=extra_body)
        refuse_other_decision_formats(model, files)
        if is_decision_repo(files):
            return DecisionModel(model, url, api_key=api_key)
        return LLM(model, base_url=url, api_key=api_key, extra_body=extra_body)
    if api_key or os.environ.get("OPENAI_BASE_URL"):
        return LLM(model, api_key=api_key, extra_body=extra_body)
    files = repo_files(model)
    if files is not None:
        return load_local(model, files)
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
