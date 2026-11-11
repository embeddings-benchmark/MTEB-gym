"""What a client of a server needs: reaching it, a request that survives a busy server, and knowing when an input does
not fit."""

from __future__ import annotations

import math
import os
import time

import httpx2

GRADES = [str(i) for i in range(6)]

READ_ONE_TOKEN = {  # the full distribution of the next token: a model's generation defaults must not truncate it
    "max_tokens": 1,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "min_p": 0.0,
    "repetition_penalty": 1.0,
    "add_special_tokens": False,
    "return_tokens_as_token_ids": True,
}


def model_card(name: str, reference: str | None = None) -> str:
    """Where a model's card says how to serve it."""
    return reference or f"https://huggingface.co/{name}"


def server_url(url: str) -> str:
    return url.rstrip("/").removesuffix("/v1")


def server_client(api_key: str | None) -> httpx2.Client:
    # a busy server queues requests, so the timeout is long
    key = api_key or os.environ.get("MTEB_GYM_API_KEY") or "EMPTY"
    return httpx2.Client(timeout=600.0, headers={"Authorization": f"Bearer {key}"})


def served_models(http: httpx2.Client, base_url: str) -> list[dict]:
    """What a server lists at /v1/models."""
    try:
        r = http.get(f"{base_url}/v1/models")
        r.raise_for_status()
        return r.json()["data"]
    except (httpx2.HTTPError, ValueError, KeyError, TypeError) as e:
        raise ValueError(f"could not reach {base_url}: {e!r}") from e


def softmax(options: list[str], logits: list[float]) -> dict[str, float]:
    largest = max(logits)
    weights = [math.exp(z - largest) for z in logits]
    return {o: w / sum(weights) for o, w in zip(options, weights, strict=True)}


class InputTooLong(Exception):
    """The input does not fit the model's context."""


def _too_long(status: int | None, message: str) -> bool:
    message = message.lower()
    return status == 413 or (
        status == 400 and ("maximum context length" in message or "context_length_exceeded" in message)
    )


def post_json(http: httpx2.Client, url: str, body: dict, max_retries: int) -> dict:
    # retries back off; a client error is final
    error = ""
    for attempt in range(max_retries + 1):
        if attempt:
            time.sleep(min(60, 2 ** (attempt - 1)))
        try:
            r = http.post(url, json=body)
        except httpx2.TransportError as e:
            error = repr(e)
            continue
        if (r.status_code >= 500 and r.status_code != 501) or r.status_code in (408, 429):
            error = f"HTTP {r.status_code}: {r.text[:200]}"
            continue
        if _too_long(r.status_code, r.text):
            raise InputTooLong(r.text[:200])
        r.raise_for_status()
        try:
            return r.json()
        except ValueError:
            error = f"not JSON: {r.text[:200]}"
    raise RuntimeError(f"{url} failed after {max_retries + 1} attempts: {error}")
