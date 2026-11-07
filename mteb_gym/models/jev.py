"""Decision models in the format of JEV-9B and JEV-27B: a repo holding `judge_config.json`, or its `adapter_vllm/` part.
No entry is needed for a new one. Prompt, one forward pass, a probability per option, as their model cards do it."""

from __future__ import annotations

import json
import math
import os
import threading
from pathlib import Path

import httpx
from huggingface_hub import hf_hub_download, snapshot_download

from ..local import fits, shared, transformers_and_peft
from ..serving import READ_ONE_TOKEN, post_json

_LETTERS = "ABCDEFGHIJKLMNOP"
_OPTIONS = {"noul": ["false", "true"], "score": [str(i) for i in range(6)]}


def _decision_head(name: str) -> dict:
    with open(hf_hub_download(name, "adapter_vllm/decision_head.json")) as f:
        return json.load(f)


def serve_command(name: str, head: dict) -> str:
    """How to serve a model of this format, by what its decision head says about it."""
    folder = name.split("/")[-1]
    download = f"hf download {name} --local-dir {folder}\n"
    if (
        head.get("prefix_token") is not None
    ):  # Gemma: vLLM needs the patch the repo holds for a LoRA on its tied lm_head
        return (
            f"{download}# apply {folder}/patches/vllm-gemma4-lm-head-lora.patch to vLLM, then:\nbash {folder}/serve.sh"
        )
    flags = (
        f"--served-model-name {name} --max-model-len 65536 --max-num-seqs 64"
        if head.get("merged")  # the decision model is in the weights: no module to add
        else f"--enable-lora --max-lora-rank 32 --lora-modules jev-decision={folder}/adapter_vllm"
    )
    return f"{download}vllm serve {folder} {flags} --logprobs-mode processed_logprobs"


def is_jev_repo(files: set[str]) -> bool:
    """A cached repo may hold only part of what is on the Hub."""
    return bool(files & {"judge_config.json", "adapter_vllm/decision_head.json"})


def decision_prompt(kind: str, state: str, question: str, options: list[str] | None) -> tuple[list[str], str]:
    """The options of a decision and its prompt, as the model cards write them."""
    options = _OPTIONS.get(kind, options)
    lines = [f"{_LETTERS[i]}) {o}" for i, o in enumerate(options)] if kind == "choice" else options
    text = f"[kind] {kind}\n[state] {state}\n[question] {question}\n[options]\n" + "\n".join(lines) + "\n[decision]:"
    return options, text


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
            head = _decision_head(self.model)
            with open(hf_hub_download(self.model, "calibration.json")) as f:
                self._temperature = json.load(f)["per_kind"]
            self._served_as = self.model if head.get("merged") else "jev-decision"
            self._serve = serve_command(self.model, head)
            r = self._http.get(f"{self.base_url}/v1/models")
            r.raise_for_status()
            if self._served_as not in {m["id"] for m in r.json()["data"]}:
                raise ValueError(f"{self.base_url} does not serve {self._served_as}. Restart it:\n{self._serve}")
            self._head = head

    def decide(self, kind: str, state: str, question: str, options: list[str] | None = None) -> dict[str, float]:
        self._prepare()
        options, text = decision_prompt(kind, state, question, options)
        start = self._head["slots"]["ranges"][kind][0]
        ids = self._head["verbalizer_ids"][start : start + len(options)]
        out = post_json(
            self._http,
            f"{self.base_url}/v1/completions",
            {
                "model": self._served_as,
                "prompt": ("<bos>" if self._head.get("prefix_token") is not None else "") + text,
                "logprobs": len(options),
                "allowed_token_ids": ids,
                **READ_ONE_TOKEN,
            },
            self.max_retries,
        )
        found = {int(k.split(":")[1]): v for k, v in out["choices"][0]["logprobs"]["top_logprobs"][0].items()}
        if not set(ids) <= set(found):
            raise ValueError(f"{self.base_url} left out options' probabilities. Restart it:\n{self._serve}")
        bias = self._head["bias"][start : start + len(options)]
        logits = [(found[i] + b) / self._temperature[kind] for i, b in zip(ids, bias, strict=True)]
        largest = max(logits)
        weights = [math.exp(z - largest) for z in logits]
        return {o: w / sum(weights) for o, w in zip(options, weights, strict=True)}


class LocalDecisionModel:
    """A decision model in this process, as its model card loads it."""

    how = "decision model"

    def __init__(self, name: str, has_adapter: bool = True, has_head: bool = True):
        self.model = name
        self._has_adapter = has_adapter
        self._has_head = has_head  # without one, the decision model is in the weights: its answers are the next token's
        self._lock = threading.Lock()
        self._net = None

    def _load(self):
        import torch
        from safetensors.torch import load_file

        peft, transformers = transformers_and_peft()
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(self.model)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        net = transformers.AutoModelForCausalLM.from_pretrained(self.model, dtype=torch.bfloat16, device_map=device)
        if self._has_adapter:
            adapter = Path(snapshot_download(self.model, allow_patterns="adapter/*")) / "adapter"
            net = peft.PeftModel.from_pretrained(net, str(adapter)).merge_and_unload()
        with open(hf_hub_download(self.model, "judge_config.json")) as f:
            config = json.load(f)
        if self._has_head:
            head = load_file(hf_hub_download(self.model, "head.safetensors"))
            self._weight, self._bias = head["proj.weight"].to(net.device), head["proj.bias"].to(net.device)
        else:
            head = _decision_head(self.model)
            if not head.get("merged"):
                command = serve_command(self.model, head)
                raise ValueError(
                    f"{self.model} is missing head.safetensors. Download it, or serve the model:\n{command}"
                )
            self._bias = torch.tensor(head["bias"], device=net.device)
            self._tokens = config["verbalizer_ids"]
        self._slots = config["slots"]["ranges"]
        self._softcap = config.get("softcap")
        has_prefix = config.get("readout", {}).get("prefix_token") is not None
        self._prefix = self._tokenizer.bos_token if has_prefix else ""
        with open(hf_hub_download(self.model, "calibration.json")) as f:
            self._temperature = json.load(f)["per_kind"]
        self._net = net.eval()

    def decide(self, kind: str, state: str, question: str, options: list[str] | None = None) -> dict[str, float]:
        import torch

        options, text = decision_prompt(kind, state, question, options)
        with self._lock, fits(self.model, lambda: serve_command(self.model, _decision_head(self.model))):
            if self._net is None:
                self._load()
            ids = self._tokenizer(self._prefix + text, return_tensors="pt", add_special_tokens=False).to(
                self._net.device
            )
            with torch.no_grad(), torch.autocast(self._net.device.type, dtype=torch.bfloat16):
                if self._has_head:
                    last = self._net.model(**ids).last_hidden_state[0, -1]
                else:
                    last = self._net(**ids, logits_to_keep=1).logits[0, -1]
        last = last.float()
        logits = self._weight @ last + self._bias if self._has_head else last[self._tokens] + self._bias
        start = self._slots[kind][0]
        if self._softcap:
            logits = self._softcap * torch.tanh(logits / self._softcap)
        logits = logits / self._temperature[kind]
        return dict(zip(options, torch.softmax(logits[start : start + len(options)], 0).tolist(), strict=True))


def load(name: str, url: str | None, api_key: str | None, files: set[str]):
    """Served at `url`, or loaded here on first use."""
    if url:
        return DecisionModel(name, url, api_key=api_key)
    return shared(LocalDecisionModel, name, "adapter/adapter_config.json" in files, "head.safetensors" in files)
