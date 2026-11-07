"""Decision models in the format of JEV-9B and JEV-27B: a repo holding `judge_config.json`, or its `adapter_vllm/` part.
Prompt, one forward pass, a probability per option, as their model cards do it: on a server, or in this process."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from ..local import fetch, fetch_folder, fits, repo_files, shared, transformers_and_peft
from ..serving import GRADES, READ_ONE_TOKEN, post_json, served_models, server_client, server_url, softmax

logger = logging.getLogger(__name__)

LETTERS = "ABCDEFGHIJKLMNOP"


def decision_head(name: str, revision: str | None) -> dict:
    with open(fetch(name, "adapter_vllm/decision_head.json", revision)) as f:
        return json.load(f)


def serve_command(name: str, head: dict, revision: str | None = None) -> str:
    """How to serve a model of this format that is not listed, by what its decision head says about it."""
    folder = name.split("/")[-1]
    at = f" --revision {revision}" if revision else ""
    download = f"hf download {name}{at} --local-dir {folder}\n"
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
    options = options or GRADES
    lines = [f"{LETTERS[i]}) {o}" for i, o in enumerate(options)] if kind == "choice" else options
    text = f"[kind] {kind}\n[state] {state}\n[question] {question}\n[options]\n" + "\n".join(lines) + "\n[decision]:"
    return options, text


class JevServer:
    """A decision model on a server, by its model card's recipe: the server returns the log-probability of each option's
    token, and the head's bias and the temperature are applied here. `decide("score", state, question)` returns a
    probability for each grade "0" to "5", and `decide("choice", state, question, options)` one for each option, in one
    forward pass. Nothing is fetched until the first decision, so judgments already cached need no server."""

    how = "decision model"

    def __init__(
        self,
        name: str,
        revision: str | None = None,
        url: str | None = None,
        api_key: str | None = None,
        serve: str | None = None,
        max_retries: int = 6,
    ):
        self.model = name
        self.revision = revision
        self.base_url = server_url(url)
        self.max_retries = max_retries
        self.serve = serve
        self._http = server_client(api_key)
        self._lock = threading.Lock()
        self._head = None

    def _prepare(self) -> None:
        """The two small files of the repo, and whether the server has what they describe."""
        with self._lock:
            if self._head is not None:
                return
            head = decision_head(self.model, self.revision)
            with open(fetch(self.model, "calibration.json", self.revision)) as f:
                self._temperature = json.load(f)["per_kind"]
            self._served_as = self.model if head.get("merged") else "jev-decision"
            self.serve = self.serve or serve_command(self.model, head, self.revision)
            served = served_models(self._http, self.base_url)
            if self._served_as not in {m["id"] for m in served}:
                raise ValueError(f"{self.base_url} does not serve {self._served_as}. Restart it:\n{self.serve}")
            names = " ".join(f"{m['id']} {m.get('root')}" for m in served)
            # a renamed folder also lands here, so a warning and not an error
            if self.model.split("/")[-1].lower() not in names.lower():
                logger.warning("%s does not look like %s: it serves %s", self.base_url, self.model, names)
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
            raise ValueError(f"{self.base_url} left out options' probabilities. Restart it:\n{self.serve}")
        bias = self._head["bias"][start : start + len(options)]
        logits = [(found[i] + b) / self._temperature[kind] for i, b in zip(ids, bias, strict=True)]
        return softmax(options, logits)


class LocalDecisionModel:
    """A decision model in this process, as its model card loads it."""

    how = "decision model"

    def __init__(
        self,
        name: str,
        revision: str | None = None,
        has_adapter: bool = True,
        has_head: bool = True,
        serve: str | None = None,
    ):
        self.model = name
        self.revision = revision
        self.serve = serve
        self._has_adapter = has_adapter
        self._has_head = has_head  # without one, the decision model is in the weights: its answers are the next token's
        self._lock = threading.Lock()
        self._net = None

    def _serve_command(self) -> str:
        return self.serve or serve_command(self.model, decision_head(self.model, self.revision), self.revision)

    def _load(self):
        import torch
        from safetensors.torch import load_file

        peft, transformers = transformers_and_peft()
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(self.model, revision=self.revision)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        net = transformers.AutoModelForCausalLM.from_pretrained(
            self.model, revision=self.revision, dtype=torch.bfloat16, device_map=device
        )
        if self._has_adapter:
            adapter = Path(fetch_folder(self.model, self.revision, "adapter/*")) / "adapter"
            net = peft.PeftModel.from_pretrained(net, str(adapter)).merge_and_unload()
        with open(fetch(self.model, "judge_config.json", self.revision)) as f:
            config = json.load(f)
        if self._has_head:
            head = load_file(fetch(self.model, "head.safetensors", self.revision))
            self._weight, self._bias = head["proj.weight"].to(net.device), head["proj.bias"].to(net.device)
        else:
            head = decision_head(self.model, self.revision)
            if not head.get("merged"):
                raise ValueError(
                    f"{self.model} is missing head.safetensors. Download it, or serve the model:\n{self._serve_command()}"
                )
            self._bias = torch.tensor(head["bias"], device=net.device)
            self._tokens = config["verbalizer_ids"]
        self._slots = config["slots"]["ranges"]
        self._softcap = config.get("softcap")
        has_prefix = config.get("readout", {}).get("prefix_token") is not None
        self._prefix = self._tokenizer.bos_token if has_prefix else ""
        with open(fetch(self.model, "calibration.json", self.revision)) as f:
            self._temperature = json.load(f)["per_kind"]
        self._net = net.eval()

    def decide(self, kind: str, state: str, question: str, options: list[str] | None = None) -> dict[str, float]:
        import torch

        options, text = decision_prompt(kind, state, question, options)
        with self._lock, fits(self.model, self._serve_command):
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


def jev_loader(
    name: str,
    revision: str | None = None,
    url: str | None = None,
    api_key: str | None = None,
    serve: str | None = None,
    files: set[str] | None = None,
):
    """Served at `url`, or loaded here on first use, once per model."""
    if url:
        return JevServer(name, revision, url, api_key, serve)
    files = repo_files(name, revision) if files is None else files
    if files is None:
        raise ValueError(f"{name} at revision {revision} is not on Hugging Face or in its local cache.")
    adapter, head = "adapter/adapter_config.json" in files, "head.safetensors" in files
    return shared(LocalDecisionModel, name, revision, adapter, head, serve)
