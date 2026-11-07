"""decider-ai's models, in this process through its own package. A long document is cut to what its reader takes."""

from __future__ import annotations

import threading

from ..local import fetch_folder, fits, shared
from ..serving import GRADES

MAX_TOKENS = 3072


class DeciderModel:
    how = "decision model"

    def __init__(self, name: str, revision: str | None = None):
        self.model = name
        self.revision = revision
        self._lock = threading.Lock()
        self._net = None

    def _load(self):
        try:
            from decider.infer import Decider
        except ImportError as e:
            raise ImportError(
                f"{self.model} needs the `decider` extra: pip install 'mteb-gym[decider] @ git+https://github.com/embeddings-benchmark/MTEB-gym.git' ({e})"
            ) from e
        self._net = Decider(fetch_folder(self.model, self.revision))

    def decide(self, kind: str, state: str, question: str, options: list[str] | None = None) -> dict[str, float]:
        options = options or GRADES
        with self._lock, fits(self.model):
            if self._net is None:
                self._load()
            ask = {"question": question, "options": options, "_type": kind}
            answer = self._net.decide(state, [ask], max_ctx_tokens=MAX_TOKENS)[0]
        return dict(zip(options, answer["probs_list"], strict=True))


def decider_loader(
    name: str, revision: str | None = None, url: str | None = None, api_key: str | None = None, serve: str | None = None
):
    """Loaded here on first use, once per model. Its server grades on another level under the same name, so it has none."""
    if url:
        raise ValueError(f"{name} runs in this process through the decider-ai package. Leave out its URL.")
    return shared(DeciderModel, name, revision)
