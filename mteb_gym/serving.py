"""What a client of a server needs: a request that survives a busy server, and knowing when an input does not fit."""

from __future__ import annotations

import time

import httpx

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


class InputTooLong(Exception):
    """The input does not fit the model's context."""


def _too_long(status: int | None, message: str) -> bool:
    message = message.lower()
    return status == 413 or (
        status == 400 and ("maximum context length" in message or "context_length_exceeded" in message)
    )


def post_json(http: httpx.Client, url: str, body: dict, max_retries: int) -> dict:
    # a busy server queues requests, so the timeout is long and retries back off; a client error is final
    error = ""
    for attempt in range(max_retries + 1):
        if attempt:
            time.sleep(min(60, 2 ** (attempt - 1)))
        try:
            r = http.post(url, json=body)
        except httpx.TransportError as e:
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
