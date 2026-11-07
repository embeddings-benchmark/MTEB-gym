"""
Models loaded directly from Hugging Face into this process, on first use. A repo holding `judge_config.json` and
`head.safetensors` is a decision model: prompt, one forward pass, a probability per option. Any other is a chat model.
"""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from functools import cache

import httpx
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError

_LETTERS = "ABCDEFGHIJKLMNOP"
_OPTIONS = {"noul": ["false", "true"], "score": [str(i) for i in range(6)]}
_MAX_NEW_TOKENS = 1024


def repo_files(name: str) -> set[str] | None:
    """The files of a Hugging Face model, or None if no such model can be reached."""
    try:
        return {f.rfilename for f in HfApi().model_info(name).siblings}
    except (RepositoryNotFoundError, HFValidationError, httpx.TransportError, OSError):
        return None


@contextmanager
def _fits(name: str):
    """An out-of-memory error from a model loaded here says what to do about it."""
    import torch

    try:
        yield
    except torch.cuda.OutOfMemoryError as e:
        raise torch.cuda.OutOfMemoryError(
            f"{name} does not fit in GPU memory. Serve it and pass judge_url or generator_url."
        ) from e


def _imports():
    try:
        import peft
        import transformers
    except ImportError as e:
        raise ImportError(
            f"loading a model directly needs the `local` extra: pip install 'mteb-gym[local]' ({e})"
        ) from e
    return peft, transformers


class LocalDecisionModel:
    """A decision model in this process, as its model card loads it."""

    def __init__(self, name: str, has_adapter: bool = True):
        self.model = name
        self._has_adapter = has_adapter
        self._lock = threading.Lock()
        self._net = None

    def _load(self):
        import torch
        from safetensors.torch import load_file

        peft, transformers = _imports()
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(self.model)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        net = transformers.AutoModelForCausalLM.from_pretrained(self.model, dtype=torch.bfloat16, device_map=device)
        if self._has_adapter:
            net = peft.PeftModel.from_pretrained(net, self.model, subfolder="adapter").merge_and_unload()
        head = load_file(hf_hub_download(self.model, "head.safetensors"))
        self._weight, self._bias = head["proj.weight"].to(net.device), head["proj.bias"].to(net.device)
        with open(hf_hub_download(self.model, "judge_config.json")) as f:
            self._slots = json.load(f)["slots"]["ranges"]
        with open(hf_hub_download(self.model, "calibration.json")) as f:
            self._temperature = json.load(f)["per_kind"]
        self._net = net.eval()

    def decide(self, kind: str, state: str, question: str, options: list[str] | None = None) -> dict[str, float]:
        import torch

        options = _OPTIONS.get(kind, options)
        lines = [f"{_LETTERS[i]}) {o}" for i, o in enumerate(options)] if kind == "choice" else options
        text = (
            f"[kind] {kind}\n[state] {state}\n[question] {question}\n[options]\n" + "\n".join(lines) + "\n[decision]:"
        )
        with self._lock, _fits(self.model):
            if self._net is None:
                self._load()
            ids = self._tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self._net.device)
            with torch.no_grad(), torch.autocast(self._net.device.type, dtype=torch.bfloat16):
                last = self._net.model(**ids).last_hidden_state[0, -1].float()
        start = self._slots[kind][0]
        logits = (self._weight @ last + self._bias) / self._temperature[kind]
        return dict(zip(options, torch.softmax(logits[start : start + len(options)], 0).tolist(), strict=True))


class LocalChatModel:
    """A chat model in this process, at the model's own generation settings, with reasoning off."""

    def __init__(self, name: str):
        self.model = name
        self._lock = threading.Lock()
        self._net = None

    def _load(self):
        import torch

        _, transformers = _imports()
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(self.model)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._net = transformers.AutoModelForCausalLM.from_pretrained(
            self.model, dtype=torch.bfloat16, device_map=device
        ).eval()

    def chat(self, messages: list[dict]) -> str:
        import torch

        with self._lock, _fits(self.model):
            if self._net is None:
                self._load()
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            ids = self._tokenizer(text, return_tensors="pt").to(self._net.device)
            with torch.no_grad():
                out = self._net.generate(
                    **ids,
                    max_new_tokens=_MAX_NEW_TOKENS,
                    do_sample=True,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
        return self._tokenizer.decode(out[0, ids["input_ids"].shape[1] :], skip_special_tokens=True)


@cache
def _shared(cls, name: str, *args):
    """One object per model: a judge and a generator of one name share it, and so do repeated runs."""
    return cls(name, *args)


def load_local(name: str, files: set[str]):
    """The model holding `files`, which is loaded only when first asked something."""
    if "judge_config.json" in files and "head.safetensors" in files:
        return _shared(LocalDecisionModel, name, "adapter/adapter_config.json" in files)
    return _shared(LocalChatModel, name)
