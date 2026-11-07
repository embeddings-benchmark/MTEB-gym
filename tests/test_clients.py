"""The clients' behaviour against a fake server."""

import json
import math
from types import SimpleNamespace as NS

import httpx
import huggingface_hub.errors as errors
import pytest
import torch

from mteb_gym.llm import LLM, llm_settings
from mteb_gym.models.jev import DecisionModel


def decision_model(handler, **kw):
    """A decision model that has read its repo: `noul`, tokens 100 and 101, no bias, temperature 1."""
    model = DecisionModel("m", "http://server/v1", max_retries=2, **kw)
    model._http = httpx.Client(transport=httpx.MockTransport(handler), headers=model._http.headers)
    model._head = {"bias": [0.0, 0.0], "verbalizer_ids": [100, 101], "slots": {"ranges": {"noul": [0, 2]}}}
    model._temperature, model._served_as = {"noul": 1.0}, "jev-decision"
    return model


EVEN = {"choices": [{"logprobs": {"top_logprobs": [{"token_id:100": -1.0, "token_id:101": -1.0}]}}]}
HALF = {"false": 0.5, "true": 0.5}


def test_decide_retries_server_errors_then_gives_up(monkeypatch):
    monkeypatch.setattr("mteb_gym.serving.time.sleep", lambda s: None)
    calls = []

    def fails_twice(request):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, json=EVEN)

    assert decision_model(fails_twice).decide("noul", "s", "q") == HALF and len(calls) == 3
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        decision_model(lambda r: httpx.Response(500, text="down")).decide("noul", "s", "q")


def test_decide_does_not_retry_a_client_error():
    calls = []

    def rejected(request):
        calls.append(1)
        return httpx.Response(401, request=request)

    with pytest.raises(httpx.HTTPStatusError):
        decision_model(rejected).decide("noul", "s", "q")
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
    monkeypatch.setattr("mteb_gym.serving.time.sleep", lambda s: None)
    answers = iter([httpx.Response(status), httpx.Response(200, json=EVEN)])
    assert decision_model(lambda r: next(answers)).decide("noul", "s", "q") == HALF


def test_decide_reports_a_reply_that_is_not_json(monkeypatch):
    monkeypatch.setattr("mteb_gym.serving.time.sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="not JSON"):
        decision_model(lambda r: httpx.Response(200, text="<html>")).decide("noul", "s", "q")


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
        decision_model(lambda r: httpx.Response(400, text=too_long)).decide("noul", "s", "q")
    with pytest.raises(httpx.HTTPStatusError):
        decision_model(lambda r: httpx.Response(400, text="unknown kind")).decide("noul", "s", "q")


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
    local.shared.cache_clear()
    return files


JEV = {"judge_config.json", "head.safetensors", "calibration.json", "adapter/adapter_config.json"}


def test_a_decision_model_on_hugging_face_is_told_by_its_files(hub):
    from mteb_gym.judge import is_decision_model
    from mteb_gym.llm import load_model
    from mteb_gym.local import LocalChatModel
    from mteb_gym.models.jev import LocalDecisionModel

    hub["org/judge"], hub["org/merged"] = JEV, {"judge_config.json", "head.safetensors"}
    hub["org/chat"] = {"model.safetensors"}
    judge, merged = load_model("org/judge"), load_model("org/merged")
    assert isinstance(judge, LocalDecisionModel) and judge._has_adapter and not merged._has_adapter
    assert is_decision_model(judge) and not is_decision_model(load_model("org/chat"))
    assert isinstance(load_model("org/chat"), LocalChatModel)


def test_one_name_is_one_model(hub):
    from mteb_gym.llm import load_model

    hub["org/chat"] = {"model.safetensors"}
    assert load_model("org/chat") is load_model("org/chat")


def test_a_name_with_a_key_never_asks_hugging_face(monkeypatch, hub):
    from mteb_gym import llm

    monkeypatch.setattr(llm, "repo_files", lambda name: pytest.fail("no need to ask"))
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
    bos_token = "<bos>"

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
    from mteb_gym.models import jev

    weight = torch.tensor([[-1.0, 7.0], [math.log(3) - 2, 7.0], [-1.0, 7.0], [math.log(2) - 1, 7.0], [99.0, 7.0]])
    save_file(
        {"proj.weight": weight, "proj.bias": torch.tensor([1.0, 2.0, 1.0, 1.0, 1.0])}, tmp_path / "head.safetensors"
    )
    (tmp_path / "judge_config.json").write_text(json.dumps({"slots": {"ranges": {"noul": [0, 2], "choice": [2, 5]}}}))
    (tmp_path / "calibration.json").write_text(json.dumps({"per_kind": {"noul": 2.0, "choice": 1.0}}))
    (tmp_path / "adapter_vllm").mkdir()
    (tmp_path / "adapter_vllm/decision_head.json").write_text(json.dumps({"bias": [0.0] * 5}))
    tokenizer, net, built, kwargs = _Tokenizer(), _Net(), [], {}

    def merged():
        net.merged = True
        return net

    transformers = NS(
        AutoTokenizer=NS(from_pretrained=lambda name: tokenizer),
        AutoModelForCausalLM=NS(from_pretrained=lambda name, **kw: kwargs.update(kw) or built.append(name) or net),
    )
    peft = NS(PeftModel=NS(from_pretrained=lambda base, path: NS(merge_and_unload=merged)))
    monkeypatch.setattr(jev, "snapshot_download", lambda name, allow_patterns: str(tmp_path))
    monkeypatch.setattr(local, "transformers_and_peft", lambda: (peft, transformers))
    monkeypatch.setattr(jev, "transformers_and_peft", lambda: (peft, transformers))
    monkeypatch.setattr(jev, "hf_hub_download", lambda name, file: str(tmp_path / file))
    return NS(tokenizer=tokenizer, net=net, built=built, kwargs=kwargs, transformers=transformers, folder=tmp_path)


def test_a_decision_is_the_softmax_of_its_slots_at_its_temperature(fake_repo):
    from mteb_gym.models.jev import LocalDecisionModel

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
    from mteb_gym.models.jev import LocalDecisionModel

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
    from mteb_gym.local import LocalChatModel
    from mteb_gym.models.jev import LocalDecisionModel

    monkeypatch.setattr(torch.cuda, "is_available", lambda: gpu)
    for model in (LocalDecisionModel("org/a", False), LocalChatModel("org/b")):
        fake_repo.kwargs.clear()
        model._load()
        assert fake_repo.kwargs["device_map"] == device


def _ask_decision(name):
    from mteb_gym.models.jev import LocalDecisionModel

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
    said = str(e.value)
    assert said.startswith("org/big does not fit in GPU memory. Serve it, then pass judge_url or generator_url:\n")
    assert "vllm serve" in said and "org/big" in said.split("\n", 1)[1]
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
    from mteb_gym.local import LocalChatModel
    from mteb_gym.models.jev import LocalDecisionModel

    assert model_id(LocalDecisionModel("org/m")) == model_id(DecisionModel("org/m", "http://x")) == "org/m"
    assert model_id(LocalChatModel("org/m")) == model_id(LLM("org/m", base_url="http://x", api_key="k")) == "org/m"


def test_a_missing_extra_says_which_to_install(monkeypatch):
    import sys

    from mteb_gym import local

    monkeypatch.setitem(sys.modules, "peft", None)
    with pytest.raises(ImportError, match=r"mteb-gym\[local\]"):
        local.transformers_and_peft()


def test_a_decision_model_with_a_prefix_and_a_softcap_applies_both(fake_repo):
    from mteb_gym.models.jev import LocalDecisionModel

    config = {"slots": {"ranges": {"noul": [0, 2]}}, "softcap": 30.0, "readout": {"prefix_token": 2}}
    (fake_repo.folder / "judge_config.json").write_text(json.dumps(config))
    p = LocalDecisionModel("org/gemma", False).decide("noul", "s", "q")
    assert fake_repo.tokenizer.seen[-1][0].startswith("<bos>[kind] noul")
    # logits 0 and log 3, capped to 30 tanh(log 3 / 30), then halved by the temperature
    odds = math.exp(30 * math.tanh(math.log(3) / 30) / 2)
    assert p["true"] == pytest.approx(odds / (1 + odds))


@pytest.fixture
def server(monkeypatch, hub, tmp_path):
    """A plain vLLM server and the repo's two small files. A completion gives each allowed token -1, except 109 at -2."""
    from mteb_gym.models import jev

    bias = [9.0] * 24
    bias[0], bias[1], bias[8], bias[9] = 0.0, 1.0, math.log(2), 0.0
    head = {
        "bias": bias,
        "verbalizer_ids": list(range(100, 124)),
        "slots": {"ranges": {"noul": [0, 2], "choice": [8, 24]}},
    }
    s = NS(head=head, requests=[], http=[], downloads=[], models=["base", "jev-decision"], logprobs=None)

    def files(name, file):
        s.downloads.append(file)
        path = tmp_path / file.replace("/", "_")
        path.write_text(
            json.dumps(s.head if file.endswith("decision_head.json") else {"per_kind": {"noul": 2.0, "choice": 1.0}})
        )
        return str(path)

    def handle(request):
        s.http.append(request)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": m} for m in s.models]})
        body = json.loads(request.content)
        s.requests.append(body)
        found = s.logprobs or {f"token_id:{t}": -2.0 if t == 109 else -1.0 for t in body["allowed_token_ids"]}
        return httpx.Response(200, json={"choices": [{"logprobs": {"top_logprobs": [found]}}]})

    client = httpx.Client
    monkeypatch.setattr(httpx, "get", lambda url, **kw: handle(httpx.Request("GET", url)))
    monkeypatch.setattr(jev, "hf_hub_download", files)
    monkeypatch.setattr(httpx, "Client", lambda **kw: client(transport=httpx.MockTransport(handle), **kw))
    hub["org/jev"] = JEV | {"adapter_vllm/decision_head.json"}
    hub["org/chat"] = {"model.safetensors"}
    hub["org/v11"] = {"decision_config.json", "lm_head.safetensors"}
    hub["org/decider"] = {"decider_config.json", "model.safetensors"}
    return s


def test_a_served_decision_model_fetches_nothing_until_it_is_asked_and_then_once(server):
    from mteb_gym.llm import load_model

    model = load_model("org/jev", "http://server/v1", api_key="k")
    assert server.downloads == [] and server.http == []
    model.decide("noul", "s", "q")
    model.decide("noul", "s", "q")
    assert server.downloads == ["adapter_vllm/decision_head.json", "calibration.json"]
    assert [r.url.path for r in server.http] == ["/v1/models", "/v1/completions", "/v1/completions"]
    assert server.http[0].headers["authorization"] == "Bearer k"


def test_a_decision_repo_on_a_plain_server_follows_the_cards_recipe(server):
    from mteb_gym.judge import is_decision_model
    from mteb_gym.llm import load_model

    model = load_model("org/jev", "http://server/v1")
    assert isinstance(model, DecisionModel) and is_decision_model(model)
    # noul: (-1 + 0) / 2 and (-1 + 1) / 2, so 1 to e^(1/2)
    root = math.exp(0.5)
    assert model.decide("noul", "s", "q") == pytest.approx({"false": 1 / (1 + root), "true": root / (1 + root)})
    assert server.requests[-1] == {
        "model": "jev-decision",
        "prompt": "[kind] noul\n[state] s\n[question] q\n[options]\nfalse\ntrue\n[decision]:",
        "logprobs": 2,
        "allowed_token_ids": [100, 101],
        "max_tokens": 1,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "add_special_tokens": False,
        "return_tokens_as_token_ids": True,
    }
    # choice starts at slot 8: -1 + log 2 and -2 + 0 at temperature 1, so 2e to 1
    assert model.decide("choice", "s", "q", ["x", "y"]) == pytest.approx(
        {"x": 2 * math.e / (2 * math.e + 1), "y": 1 / (2 * math.e + 1)}
    )
    assert server.requests[-1]["allowed_token_ids"] == [108, 109]
    assert server.requests[-1]["prompt"].endswith("[options]\nA) x\nB) y\n[decision]:")


def test_a_served_decision_model_is_identified_by_its_name_alone(server):
    from mteb_gym.cache import model_id
    from mteb_gym.llm import llm_settings, load_model

    model = load_model("org/jev", "http://server/v1")
    assert model_id(model) == "org/jev"
    assert llm_settings(model) == {"model": "org/jev", "served_model": None}  # what a record holds


def test_a_plain_server_without_the_decision_module_says_which_flags_to_add(server):
    from mteb_gym.llm import load_model

    server.models = ["base"]
    with pytest.raises(
        ValueError,
        match=r"(?s)does not serve jev-decision.*hf download org/jev --local-dir jev\nvllm serve jev --enable-lora "
        r".*--lora-modules jev-decision=jev/adapter_vllm --logprobs-mode processed_logprobs",
    ):
        load_model("org/jev", "http://server/v1").decide("noul", "s", "q")


def test_a_gemma_decision_model_says_vllm_needs_the_patch_in_its_repo(server):
    from mteb_gym.llm import load_model

    server.head["prefix_token"] = 2
    server.models = ["base"]
    with pytest.raises(ValueError, match=r"(?s)patches/vllm-gemma4-lm-head-lora.patch.*bash jev/serve.sh"):
        load_model("org/jev", "http://server/v1").decide("noul", "s", "q")


def test_a_merged_decision_model_is_asked_under_its_own_name(server):
    from mteb_gym.llm import load_model

    server.head["merged"] = True
    server.models = ["other"]
    with pytest.raises(
        ValueError, match=r"(?s)does not serve org/jev.*vllm serve jev --served-model-name org/jev .*processed_logprobs"
    ):
        load_model("org/jev", "http://server/v1").decide("noul", "s", "q")
    server.models = ["org/jev"]
    load_model("org/jev", "http://server/v1").decide("noul", "s", "q")
    assert server.requests[-1]["model"] == "org/jev"


def test_a_repo_that_names_a_prefix_token_has_the_prefix_in_its_prompt(server):
    from mteb_gym.llm import load_model

    server.head["prefix_token"] = 2
    load_model("org/jev", "http://server/v1").decide("noul", "s", "q")
    assert server.requests[-1]["prompt"].startswith("<bos>[kind] noul")


def test_probabilities_the_server_left_out_are_an_error_and_not_a_guess(server):
    from mteb_gym.llm import load_model

    server.logprobs = {"token_id:100": -1.0, "token_id:5": -0.5}
    with pytest.raises(ValueError, match="processed_logprobs"):
        load_model("org/jev", "http://server/v1").decide("noul", "s", "q")


@pytest.mark.parametrize("name", ["org/chat", "not/on-the-hub"])
def test_a_plain_server_with_any_other_name_is_a_chat_model(server, name):
    from mteb_gym.llm import load_model

    server.models = ["base"]
    assert isinstance(load_model(name, "http://server/v1"), LLM)


@pytest.mark.parametrize("name", ["org/v11", "org/decider"])
@pytest.mark.parametrize("url", ["http://server/v1", None])
def test_a_decision_format_not_read_here_is_refused_and_not_used_for_chat(server, name, url):
    from mteb_gym.llm import load_model

    with pytest.raises(ValueError, match=rf"{name} is a decision model in a format not supported yet.*decide\(\)"):
        load_model(name, url)


def test_a_decision_repo_without_a_head_loads_as_a_decision_model_and_not_a_chat_model(hub):
    from mteb_gym.llm import load_model
    from mteb_gym.models.jev import LocalDecisionModel

    hub["org/merged"] = {"judge_config.json", "adapter_vllm/decision_head.json"}
    model = load_model("org/merged")
    assert isinstance(model, LocalDecisionModel) and not model._has_head and not model._has_adapter


def test_a_decision_in_the_weights_is_read_from_the_next_token(fake_repo, monkeypatch):
    from mteb_gym.models.jev import LocalDecisionModel

    config = {"slots": {"ranges": {"noul": [0, 2], "choice": [2, 5]}}, "verbalizer_ids": [10, 11, 12, 13, 14]}
    (fake_repo.folder / "judge_config.json").write_text(json.dumps(config))
    (fake_repo.folder / "adapter_vllm/decision_head.json").write_text(
        json.dumps({"merged": True, "bias": [0.0, 1.0, 1.0, 0.0, 9.0]})
    )
    logits = torch.zeros(20)
    logits[11], logits[13], logits[14] = math.log(3) - 1, math.log(2), 99.0
    asked = {}

    def called(**kw):
        asked.update(kw)
        return NS(logits=logits[None, None])

    fake_repo.net.__class__ = type("Net", (_Net,), {"__call__": lambda self, **kw: called(**kw)})
    model = LocalDecisionModel("org/merged", False, False)
    # noul: logits 0 and log 3 after the bias, at temperature 2. The other tokens are not its options.
    root = math.sqrt(3)
    assert model.decide("noul", "s", "q") == pytest.approx({"false": 1 / (1 + root), "true": root / (1 + root)})
    assert asked["logits_to_keep"] == 1
    # choice starts at slot 2, whose token 12 has bias 1 and token 13 bias 0: logits 1 and log 2
    expected = {"x": math.e / (math.e + 2), "y": 2 / (math.e + 2)}
    assert model.decide("choice", "s", "q", ["x", "y"]) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("name", "url", "said"),
    [
        ("org/jev", None, "judge: org/jev (decision model)\n"),
        ("org/chat", None, "judge: org/chat (chat model)\n"),
        ("org/jev", "http://server/v1", "judge: org/jev (decision model)\n"),
        ("org/chat", "http://server/v1", "judge: org/chat (chat model)\n"),
    ],
)
def test_a_run_says_what_each_name_turned_out_to_be(server, capsys, name, url, said):
    from mteb_gym.run import _load

    _load("judge", name, url)
    assert said in capsys.readouterr().out


@pytest.fixture
def offline_cache(monkeypatch, tmp_path):
    """No Hub to ask, and a local cache that holds one decision model and one chat model."""
    import huggingface_hub.constants as constants

    from mteb_gym import llm, local

    def offline(*a, **k):
        raise errors.OfflineModeIsEnabled("offline")

    monkeypatch.setattr(local.HfApi, "model_info", offline)
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(llm, "repo_files", local.repo_files)
    for name, files in {
        "org--jev": ["judge_config.json", "adapter_vllm/decision_head.json"],
        "org--chat": ["config.json"],
    }.items():
        folder = tmp_path / f"models--{name}"
        (folder / "refs").mkdir(parents=True)
        (folder / "refs" / "main").write_text("abc")
        for file in files:
            (folder / "snapshots" / "abc" / file).parent.mkdir(parents=True, exist_ok=True)
            (folder / "snapshots" / "abc" / file).write_text("{}")


def test_offline_a_repo_in_the_local_cache_is_read_from_it(server, offline_cache):
    from mteb_gym.local import repo_files

    assert repo_files("org/jev") == {"judge_config.json", "adapter_vllm/decision_head.json"}
    assert repo_files("org/chat") == {"config.json"}
    assert repo_files("org/absent") is None
    assert repo_files("not a repo name") is None


def test_offline_a_cached_decision_repo_is_still_a_decision_model(server, offline_cache):
    from mteb_gym.llm import load_model
    from mteb_gym.local import LocalChatModel

    assert isinstance(load_model("org/jev", "http://server/v1"), DecisionModel)
    assert isinstance(load_model("org/chat", "http://server/v1"), LLM)
    assert isinstance(load_model("org/chat"), LocalChatModel)


def test_a_partly_cached_decision_repo_is_never_a_chat_model(hub):
    from mteb_gym.llm import load_model

    hub["org/part"] = {"adapter_vllm/decision_head.json", "calibration.json"}
    from mteb_gym.models.jev import LocalDecisionModel

    assert isinstance(load_model("org/part", "http://server/v1"), DecisionModel)
    assert isinstance(load_model("org/part"), LocalDecisionModel)


def test_a_repo_missing_its_head_without_being_merged_says_so_and_does_not_read_the_next_token(fake_repo):
    from mteb_gym.models.jev import LocalDecisionModel

    (fake_repo.folder / "adapter_vllm/decision_head.json").write_text(json.dumps({"bias": [0.0] * 5}))
    with pytest.raises(ValueError, match="(?s)org/part is missing head.safetensors.*vllm serve part"):
        LocalDecisionModel("org/part", False, False).decide("noul", "s", "q")


def test_offline_a_repo_not_in_the_cache_says_so(monkeypatch, offline_cache):
    from mteb_gym.llm import load_model

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    with pytest.raises(ValueError, match="'org/absent' was not found on Hugging Face or in its local cache"):
        load_model("org/absent")


@pytest.mark.parametrize("name", ["JEV-9B", "org/absent"])
def test_a_name_that_is_no_repo_on_a_server_with_the_decision_module_is_refused(server, name):
    from mteb_gym.llm import load_model

    with pytest.raises(ValueError, match=rf"serves a decision model, but '{name}' cannot be read.*repo name"):
        load_model(name, "http://server/v1")
    server.models = ["base"]
    assert isinstance(load_model(name, "http://server/v1"), LLM)  # no module: nothing says it is a decision model


@pytest.mark.parametrize(
    "reply",
    [httpx.Response(404), httpx.Response(200, text="<html>"), httpx.Response(200, json={"models": []})],
    ids=["no route", "not json", "other shape"],
)
def test_a_server_that_cannot_list_its_models_is_treated_as_a_chat_server(monkeypatch, reply):
    from mteb_gym.llm import load_model

    monkeypatch.setattr(httpx, "get", lambda url, **kw: reply)
    assert isinstance(load_model("name", "http://server/v1"), LLM)
