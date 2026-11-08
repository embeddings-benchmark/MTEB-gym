"""
Chat models loaded directly from Hugging Face into this process, on first use, and what loading any model here needs.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from functools import cache
from pathlib import Path

import httpx
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.errors import HFValidationError, LocalEntryNotFoundError, RepositoryNotFoundError

_MAX_NEW_TOKENS = 1024


def repo_files(name: str, revision: str | None = None) -> set[str] | None:
    """The files of a Hugging Face model, from the local cache if the Hub cannot tell, or None if neither has it."""
    try:
        return {f.rfilename for f in HfApi().model_info(name, revision=revision).siblings}
    except (RepositoryNotFoundError, HFValidationError, httpx.TransportError, OSError):
        pass
    try:
        folder = Path(snapshot_download(name, revision=revision, local_files_only=True))
    except (LocalEntryNotFoundError, HFValidationError):
        return None
    return {str(path.relative_to(folder)) for path in folder.rglob("*") if path.is_file()}


@contextmanager
def _says_what_is_missing(name: str, revision: str | None):
    try:
        yield
    except LocalEntryNotFoundError as e:
        at = f" at revision {revision}" if revision else ""
        raise ValueError(f"{name}{at} is not in the local cache and cannot be downloaded: {e}") from e


def fetch(name: str, file: str, revision: str | None = None) -> str:
    """The path of one file of a repo."""
    with _says_what_is_missing(name, revision):
        return hf_hub_download(name, file, revision=revision)


def fetch_folder(name: str, revision: str | None = None, allow_patterns: str | None = None) -> str:
    """The folder of a repo, or of the files of it that match."""
    with _says_what_is_missing(name, revision):
        return snapshot_download(name, revision=revision, allow_patterns=allow_patterns)


@contextmanager
def fits(name: str, card: str | None = None):
    """An out-of-memory error from a model loaded here says what to do about it."""
    import torch

    try:
        yield
    except torch.cuda.OutOfMemoryError as e:
        advice = (
            f"Serve it as its model card describes, then pass judge_url or generator_url: {card}"
            if card
            else "It needs a larger GPU."
        )
        raise torch.cuda.OutOfMemoryError(f"{name} does not fit in GPU memory. {advice}") from e


def transformers_and_peft():
    try:
        import peft
        import transformers
    except ImportError as e:
        raise ImportError(
            f'loading a model directly needs the `local` extra: pip install "mteb-gym[local] @ git+https://github.com/embeddings-benchmark/MTEB-gym.git" ({e})'
        ) from e
    return peft, transformers


class LocalChatModel:
    """A chat model in this process, at the model's own generation settings, with reasoning off."""

    how = "chat model"

    def __init__(self, name: str):
        self.model = name
        self._lock = threading.Lock()
        self._net = None

    def _load(self):
        import torch

        _, transformers = transformers_and_peft()
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(self.model)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._net = transformers.AutoModelForCausalLM.from_pretrained(
            self.model, dtype=torch.bfloat16, device_map=device
        ).eval()

    def load(self) -> None:
        """Load the model now rather than on its first answer."""
        with self._lock, fits(self.model, f"https://huggingface.co/{self.model}"):
            if self._net is None:
                self._load()

    def chat(self, messages: list[dict]) -> str:
        import torch

        self.load()
        with self._lock, fits(self.model, f"https://huggingface.co/{self.model}"):
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


def load_now(client) -> None:
    """A model loaded in this process loads before a stage's progress starts, so its bar times only the work."""
    load = getattr(client, "load", None)
    if callable(load):
        load()


@cache
def shared(cls, name: str, *args):
    """One object per model: a judge and a generator of one name share it, and so do repeated runs."""
    return cls(name, *args)
