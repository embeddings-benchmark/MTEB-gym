"""The clients' behaviour against a fake server."""

import json

import httpx
import pytest

from mteb_gym.llm import LLM, DecisionModel, llm_settings


def decision_model(handler, **kw):
    model = DecisionModel("m", "http://server/v1", max_retries=2, **kw)
    model._http = httpx.Client(transport=httpx.MockTransport(handler), headers=model._http.headers)
    return model


def test_decide_pairs_each_option_with_its_probability():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = request.content
        return httpx.Response(200, json={"options": ["0", "1"], "probabilities": [0.25, 0.75]})

    assert decision_model(handler).decide("score", "state", "question") == {"0": 0.25, "1": 0.75}
    assert seen["url"] == "http://server/v1/decide" and b'"kind":"score"' in seen["body"].replace(b" ", b"")


def test_decide_retries_server_errors_then_gives_up(monkeypatch):
    monkeypatch.setattr("mteb_gym.llm.time.sleep", lambda s: None)
    calls = []

    def fails_twice(request):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, json={"options": ["1"], "probabilities": [1.0]})

    assert decision_model(fails_twice).decide("score", "s", "q") == {"1": 1.0} and len(calls) == 3
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        decision_model(lambda r: httpx.Response(500, text="down")).decide("score", "s", "q")


def test_decide_does_not_retry_a_client_error():
    calls = []

    def rejected(request):
        calls.append(1)
        return httpx.Response(401, request=request)

    with pytest.raises(httpx.HTTPStatusError):
        decision_model(rejected).decide("score", "s", "q")
    assert len(calls) == 1


def test_openai_key_is_not_sent_to_other_endpoints(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.delenv("MTEB_GYM_API_KEY", raising=False)
    assert LLM("m", base_url="http://other/v1").client.api_key == "EMPTY"
    assert DecisionModel("m", "http://other")._http.headers["authorization"] == "Bearer EMPTY"
    assert LLM("m").client.api_key == "secret"
    monkeypatch.setenv("MTEB_GYM_API_KEY", "mine")
    assert LLM("m", base_url="http://other/v1").client.api_key == "mine"
    assert DecisionModel("m", "http://other")._http.headers["authorization"] == "Bearer mine"
    assert DecisionModel("m", "http://other", api_key="given")._http.headers["authorization"] == "Bearer given"


def test_the_record_leaves_out_a_private_endpoint():
    assert "private" not in json.dumps(llm_settings(LLM("m", base_url="http://private:8000/v1", api_key="k")))


@pytest.mark.parametrize("status", [408, 429])
def test_decide_waits_out_a_busy_server(monkeypatch, status):
    monkeypatch.setattr("mteb_gym.llm.time.sleep", lambda s: None)
    answers = iter([httpx.Response(status), httpx.Response(200, json={"options": ["1"], "probabilities": [1.0]})])
    assert decision_model(lambda r: next(answers)).decide("score", "s", "q") == {"1": 1.0}


def test_decide_reports_a_reply_that_is_not_json(monkeypatch):
    monkeypatch.setattr("mteb_gym.llm.time.sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="not JSON"):
        decision_model(lambda r: httpx.Response(200, text="<html>")).decide("score", "s", "q")


def test_an_answer_cut_off_by_the_token_limit_is_empty():
    import types

    choice = types.SimpleNamespace(finish_reason="length", message=types.SimpleNamespace(content='{"winner": "A"}'))
    llm = LLM("m", api_key="x")

    def create(**kw):
        return types.SimpleNamespace(model="m", choices=[choice])

    llm.client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    assert llm.chat([]) == ""


def test_a_model_context_error_is_told_apart_from_other_bad_requests():
    from mteb_gym.llm import InputTooLong

    too_long = "This model's maximum context length is 32768 tokens. However, your prompt contains more."
    with pytest.raises(InputTooLong):
        decision_model(lambda r: httpx.Response(400, text=too_long)).decide("score", "s", "q")
    with pytest.raises(httpx.HTTPStatusError):
        decision_model(lambda r: httpx.Response(400, text="unknown kind")).decide("score", "s", "q")
