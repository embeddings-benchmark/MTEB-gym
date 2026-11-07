"""The clients' behaviour against a fake server."""

import json
import math
from types import SimpleNamespace as NS

import httpx
import huggingface_hub.errors as errors
import pytest
import torch

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


def test_an_llm_context_error_reads_as_too_long():
    from mteb_gym.llm import InputTooLong

    class Refused(Exception):
        status_code = 400

    class Completions:
        def create(self, **kw):
            raise Refused("This model's maximum context length is 32768 tokens.")

    llm = LLM("m", base_url="http://server/v1", api_key="k")
    llm.client = type("C", (), {"chat": type("H", (), {"completions": Completions()})()})()
    with pytest.raises(InputTooLong):
        llm.chat([{"role": "user", "content": "x"}])


@pytest.mark.parametrize(("status", "kind"), [(404, LLM), (422, DecisionModel), (400, DecisionModel)])
def test_a_served_model_is_told_apart_by_its_decide_endpoint(monkeypatch, status, kind):
    from mteb_gym.llm import load_model

    asked = {}

    def post(url, **kw):
        asked.update(url=url, **kw)
        return httpx.Response(status)

    monkeypatch.setattr(httpx, "post", post)
    model = load_model("m", "http://server/v1", api_key="k")
    assert isinstance(model, kind) and model.model == "m"
    assert asked["url"] == "http://server/v1/decide" and asked["headers"]["Authorization"] == "Bearer k"


def test_a_name_without_a_url_is_an_api_model_and_makes_no_call(monkeypatch):
    from mteb_gym.llm import load_model

    monkeypatch.setattr(httpx, "post", lambda *a, **k: pytest.fail("no server to ask"))
    assert isinstance(load_model("m", api_key="k"), LLM)
    existing = LLM("m", api_key="k")
    assert load_model(existing) is existing
    assert load_model(None) is None


@pytest.fixture
def hub(monkeypatch):
    """What Hugging Face holds, and how much GPU memory is free, without either."""
    import mteb_gym.local as local
    from mteb_gym import llm

    files = {}
    monkeypatch.setattr(llm, "repo_files", files.get)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    local._shared.cache_clear()
    return files


JEV = {"judge_config.json", "head.safetensors", "calibration.json", "adapter/adapter_config.json"}


def test_a_decision_model_on_hugging_face_is_told_by_its_files(hub):
    from mteb_gym.judge import is_decision_model
    from mteb_gym.llm import load_model
    from mteb_gym.local import LocalChatModel, LocalDecisionModel

    hub["org/judge"], hub["org/merged"] = JEV, {"judge_config.json", "head.safetensors"}
    hub["org/half"], hub["org/chat"] = {"judge_config.json"}, {"model.safetensors"}
    judge, merged = load_model("org/judge"), load_model("org/merged")
    assert isinstance(judge, LocalDecisionModel) and judge._has_adapter and not merged._has_adapter
    assert is_decision_model(judge) and not is_decision_model(load_model("org/chat"))
    assert isinstance(load_model("org/chat"), LocalChatModel) and isinstance(load_model("org/half"), LocalChatModel)


def test_one_name_is_one_model(hub):
    from mteb_gym.llm import load_model

    hub["org/chat"] = {"model.safetensors"}
    assert load_model("org/chat") is load_model("org/chat")


def test_a_name_with_a_url_or_a_key_never_asks_hugging_face(monkeypatch, hub):
    from mteb_gym import llm

    monkeypatch.setattr(llm, "repo_files", lambda name: pytest.fail("no need to ask"))
    monkeypatch.setattr(httpx, "post", lambda url, **kw: httpx.Response(404))
    assert isinstance(llm.load_model("org/m", "http://server/v1"), LLM)
    assert isinstance(llm.load_model("org/m", api_key="k"), LLM)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://provider/v1")
    assert isinstance(llm.load_model("org/m"), LLM)


def test_a_name_hugging_face_lacks_goes_to_the_api_or_says_what_to_do(monkeypatch, hub):
    from mteb_gym.llm import load_model

    with pytest.raises(ValueError, match=r"'gpt-x' was not found on Hugging Face.*OPENAI_API_KEY.*URL"):
        load_model("gpt-x")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    assert isinstance(load_model("gpt-x"), LLM)
    hub["org/chat"] = {"model.safetensors"}
    assert not isinstance(load_model("org/chat"), LLM)  # a key does not send a Hugging Face model to the API


@pytest.mark.parametrize(
    "error",
    [
        errors.RepositoryNotFoundError("no", response=httpx.Response(404, request=httpx.Request("GET", "http://x"))),
        errors.OfflineModeIsEnabled("offline"),
        errors.HFValidationError("not a repo id"),
        httpx.ConnectError("no network"),
        ConnectionError("no network"),
    ],
    ids=lambda e: type(e).__name__,
)
def test_a_repo_that_cannot_be_reached_has_no_files(monkeypatch, error):
    from mteb_gym import local

    def missing(*a, **k):
        raise error

    monkeypatch.setattr(local.HfApi, "model_info", missing)
    assert local.repo_files("org/m") is None


def test_a_repos_files(monkeypatch):
    from mteb_gym import local

    info = NS(siblings=[NS(rfilename="a.safetensors"), NS(rfilename="README.md")])
    monkeypatch.setattr(local.HfApi, "model_info", lambda self, name: info)
    assert local.repo_files("org/m") == {"a.safetensors", "README.md"}


class _Tokens(dict):
    def to(self, device):
        return self


class _Net:
    """The last hidden state is always [1, 0], so a head's logit is its weight's first column plus its bias."""

    def __init__(self):
        self.device = torch.device("cpu")
        self.merged = False

    def eval(self):
        return self

    def model(self, **ids):
        return NS(last_hidden_state=torch.tensor([[[9.0, 9.0], [1.0, 0.0]]]))

    def generate(self, **kwargs):
        return torch.tensor([[1, 2, 3, 4, 5]])


class _Tokenizer:
    eos_token_id = 0

    def __init__(self):
        self.seen, self.template, self.decoded = [], {}, None

    def __call__(self, text, **kwargs):
        self.seen.append((text, kwargs))
        return _Tokens(input_ids=torch.tensor([[1, 2, 3]]))

    def apply_chat_template(self, messages, **kwargs):
        self.template = kwargs
        return "prompt"

    def decode(self, ids, **kwargs):
        self.decoded = ids.tolist()
        return "written"


@pytest.fixture
def fake_repo(monkeypatch, tmp_path):
    """A repo of a decision model, with transformers and peft standing in. Returns what they were asked."""
    from safetensors.torch import save_file

    from mteb_gym import local

    weight = torch.tensor([[-1.0, 7.0], [math.log(3) - 2, 7.0], [-1.0, 7.0], [math.log(2) - 1, 7.0], [99.0, 7.0]])
    save_file(
        {"proj.weight": weight, "proj.bias": torch.tensor([1.0, 2.0, 1.0, 1.0, 1.0])}, tmp_path / "head.safetensors"
    )
    (tmp_path / "judge_config.json").write_text(json.dumps({"slots": {"ranges": {"noul": [0, 2], "choice": [2, 5]}}}))
    (tmp_path / "calibration.json").write_text(json.dumps({"per_kind": {"noul": 2.0, "choice": 1.0}}))
    tokenizer, net, built, kwargs = _Tokenizer(), _Net(), [], {}

    def merged():
        net.merged = True
        return net

    transformers = NS(
        AutoTokenizer=NS(from_pretrained=lambda name: tokenizer),
        AutoModelForCausalLM=NS(from_pretrained=lambda name, **kw: kwargs.update(kw) or built.append(name) or net),
    )
    peft = NS(PeftModel=NS(from_pretrained=lambda base, name, subfolder: NS(merge_and_unload=merged)))
    monkeypatch.setattr(local, "_imports", lambda: (peft, transformers))
    monkeypatch.setattr(local, "hf_hub_download", lambda name, file: str(tmp_path / file))
    return NS(tokenizer=tokenizer, net=net, built=built, kwargs=kwargs, transformers=transformers)


def test_a_decision_is_the_softmax_of_its_slots_at_its_temperature(fake_repo):
    from mteb_gym.local import LocalDecisionModel

    model = LocalDecisionModel("org/judge")
    # noul: logits 0 and log 3, halved by the temperature, so sqrt(3) to 1
    root = math.sqrt(3)
    assert model.decide("noul", "s", "q") == pytest.approx({"false": 1 / (1 + root), "true": root / (1 + root)})
    assert fake_repo.tokenizer.seen[-1] == (
        "[kind] noul\n[state] s\n[question] q\n[options]\nfalse\ntrue\n[decision]:",
        {"return_tensors": "pt", "add_special_tokens": False},
    )
    # choice starts at slot 2: logits 0 and log 2 at temperature 1. The 99 at slot 4 is not one of its options.
    assert model.decide("choice", "s", "q", ["x", "y"]) == pytest.approx({"x": 1 / 3, "y": 2 / 3})
    assert fake_repo.tokenizer.seen[-1][0].endswith("[options]\nA) x\nB) y\n[decision]:")
    assert fake_repo.built == ["org/judge"]  # loaded once, on first use


def test_a_decision_model_merges_its_adapter_if_it_has_one(fake_repo):
    from mteb_gym.local import LocalDecisionModel

    LocalDecisionModel("org/merged", False).decide("noul", "s", "q")
    assert not fake_repo.net.merged
    LocalDecisionModel("org/judge").decide("noul", "s", "q")
    assert fake_repo.net.merged


def test_a_chat_model_turns_reasoning_off_and_returns_only_what_it_wrote(fake_repo):
    from mteb_gym.local import LocalChatModel

    model = LocalChatModel("org/chat")
    assert fake_repo.built == []  # nothing is loaded until it is asked something
    assert model.chat([{"role": "user", "content": "hi"}]) == "written"
    assert model.chat([{"role": "user", "content": "hi"}]) == "written"
    assert fake_repo.tokenizer.template["enable_thinking"] is False
    assert fake_repo.tokenizer.decoded == [4, 5]  # the three prompt tokens are left out
    assert fake_repo.built == ["org/chat"]


@pytest.mark.parametrize(("gpu", "device"), [(True, "cuda"), (False, "cpu")])
def test_a_model_loads_whole_onto_one_device_so_running_out_of_memory_raises(monkeypatch, fake_repo, gpu, device):
    from mteb_gym.local import LocalChatModel, LocalDecisionModel

    monkeypatch.setattr(torch.cuda, "is_available", lambda: gpu)
    for model in (LocalDecisionModel("org/a", False), LocalChatModel("org/b")):
        fake_repo.kwargs.clear()
        model._load()
        assert fake_repo.kwargs["device_map"] == device


def _ask_decision(name):
    from mteb_gym.local import LocalDecisionModel

    return LocalDecisionModel(name, False).decide("noul", "s", "q")


def _ask_chat(name):
    from mteb_gym.local import LocalChatModel

    return LocalChatModel(name).chat([{"role": "user", "content": "hi"}])


@pytest.mark.parametrize("ask", [_ask_decision, _ask_chat])
@pytest.mark.parametrize("where", ["loading", "running"])
def test_running_out_of_memory_says_to_serve_the_model(monkeypatch, fake_repo, ask, where):
    def out(*a, **k):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    if where == "loading":
        monkeypatch.setattr(fake_repo.transformers.AutoModelForCausalLM, "from_pretrained", out)
    else:
        monkeypatch.setattr(fake_repo.net, "model", out)
        monkeypatch.setattr(fake_repo.net, "generate", out)
    with pytest.raises(torch.cuda.OutOfMemoryError) as e:
        ask("org/big")
    assert str(e.value) == "org/big does not fit in GPU memory. Serve it and pass judge_url or generator_url."
    assert str(e.value.__cause__) == "CUDA out of memory"


@pytest.mark.parametrize("ask", [_ask_decision, _ask_chat])
def test_any_other_error_passes_through_untouched(monkeypatch, fake_repo, ask):
    def other(*a, **k):
        raise RuntimeError("something else")

    monkeypatch.setattr(fake_repo.transformers.AutoModelForCausalLM, "from_pretrained", other)
    with pytest.raises(RuntimeError, match="^something else$"):
        ask("org/m")


def test_a_local_judge_has_the_identity_of_the_same_name_served():
    from mteb_gym.cache import model_id
    from mteb_gym.local import LocalChatModel, LocalDecisionModel

    assert model_id(LocalDecisionModel("org/m")) == model_id(DecisionModel("org/m", "http://x")) == "org/m"
    assert model_id(LocalChatModel("org/m")) == model_id(LLM("org/m", base_url="http://x", api_key="k")) == "org/m"


def test_a_missing_extra_says_which_to_install(monkeypatch):
    import sys

    from mteb_gym import local

    monkeypatch.setitem(sys.modules, "peft", None)
    with pytest.raises(ImportError, match=r"mteb-gym\[local\]"):
        local._imports()
