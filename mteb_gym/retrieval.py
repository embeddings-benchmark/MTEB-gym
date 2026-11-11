"""Retrieval is mteb.evaluate on the gym task; the gym only reads mteb's prediction file."""

from __future__ import annotations

import gc
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from .corpus import Corpus

if TYPE_CHECKING:
    from mteb.abstasks.retrieval import AbsTaskRetrieval
    from mteb.models import ModelMeta
    from mteb.models.models_protocols import EncoderProtocol


@dataclass
class Ranked:
    """One model's top-k for one query."""

    qid: str
    query: str
    doc_ids: list[str]
    doc_texts: list[str]


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def _cacheable(meta: ModelMeta, model: object) -> bool:
    """A model with one vector per text that takes time to encode."""
    late_interaction = "late-interaction" in (meta.model_type or [])
    return (
        callable(getattr(model, "encode", None))
        and "bm25" not in meta.name.lower()
        and not late_interaction
        and not _is_static(getattr(model, "model", None))
    )


def _is_static(encoder: object) -> bool:
    """A static embedding model has no attention, hence no length limit, and encodes too fast to need a cache."""
    modules = getattr(encoder, "_modules", {}).values()
    return type(encoder).__name__ == "StaticEmbedding" or any(type(m).__name__ == "StaticEmbedding" for m in modules)


def _cap_length(model: object, max_seq_length: int) -> None:
    """Cap a SentenceTransformer's length. Anything else, and a static model, is left as mteb loads it."""
    encoder = getattr(model, "model", None)
    length = getattr(encoder, "max_seq_length", None)
    if encoder is not None and length is not None and length > max_seq_length and not _is_static(encoder):
        encoder.max_seq_length = max_seq_length


def _cached(model: EncoderProtocol, folder: Path) -> EncoderProtocol:
    """The model behind mteb's embedding cache."""
    import numpy as np
    from mteb.models.cache_wrappers import CachedEmbeddingWrapper
    from mteb.models.cache_wrappers.cache_backends.numpy_cache import NumpyCache

    class ReopeningCache(NumpyCache):
        """mteb's NumpyCache opens its vectors with shape (-1, dim), which numpy rejects, so it cannot reopen them."""

        def _open(self) -> None:
            if self.vector_dim is None:
                raise ValueError("the vector dimension is required to reopen the cache")
            rows = os.path.getsize(self.vectors_file) // (4 * self.vector_dim)
            self.vectors = np.memmap(self.vectors_file, dtype="float32", mode="r+", shape=(rows, self.vector_dim))

        def _initialize_vectors_file(self) -> None:
            if self.vector_dim is not None and self.vectors_file.exists():
                self._open()
            else:
                super()._initialize_vectors_file()

        def load(self) -> None:
            self._load_dimension()
            if self.index_file.exists() and self.vectors_file.exists() and self.vector_dim is not None:
                self.hash_to_index = {str(k): int(v) for k, v in json.loads(self.index_file.read_text()).items()}
                self._open()

    # The wrapper forwards this encoder's metadata; upstream types allow missing metadata for other inputs.
    return cast("EncoderProtocol", CachedEmbeddingWrapper(model, folder, cache_backend=ReopeningCache))


def predict(
    meta: ModelMeta,
    task: AbsTaskRetrieval,
    folder: Path,
    *,
    batch_size: int = 32,
    embedding_cache: str | Path | None = None,
    max_seq_length: int | None = None,
) -> Path:
    """Run the model `meta` describes on `task` through mteb, writing mteb's prediction file into
    `folder`; skipped when the file already exists. Dense, sparse and
    late-interaction models all go through the same call.

    `embedding_cache` keeps each model's vectors in mteb's cache under <embedding_cache>, per model and cap, so a corpus
    is encoded once; it holds one vector per text, so lexical, static and late-interaction models run without it.
    `max_seq_length` caps a SentenceTransformer's length; by default a model runs as mteb loads it."""
    path = folder / f"{task.metadata.name}_predictions.json"
    if path.exists():
        return path
    import mteb
    import torch

    folder.mkdir(parents=True, exist_ok=True)
    model = meta.load_model()
    if max_seq_length is not None:
        _cap_length(model, max_seq_length)
    if embedding_cache is not None and _cacheable(meta, model):
        cap = (
            "" if max_seq_length is None else f"-max{max_seq_length}"
        )  # vectors made under one cap stay apart from another's
        model = _cached(
            cast("EncoderProtocol", model), Path(embedding_cache) / f"{slug(meta.model_name_with_experiment)}{cap}"
        )
    with tempfile.TemporaryDirectory(dir=folder) as temporary:  # the file appears whole or not at all
        mteb.evaluate(
            model,
            task,
            prediction_folder=temporary,
            cache=None,
            encode_kwargs={"batch_size": batch_size},
            show_progress_bar=False,
        )
        (Path(temporary) / path.name).replace(path)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return path


def hits(path: Path, ignore_identical_ids: bool = False) -> dict[str, dict[str, float]]:
    """What each model retrieved, from mteb's prediction file. A query that is itself a document is
    dropped from its own results where mteb's flag says so (ArguAna, FiQA), as mteb does when it scores."""
    scores: dict[str, dict[str, float]] = json.loads(path.read_text())["default"]["test"]
    if ignore_identical_ids:
        scores = {qid: {d: s for d, s in rels.items() if d != qid} for qid, rels in scores.items()}
    return scores


def top_k(path: Path, corpus: Corpus, queries: dict[str, str], k: int) -> list[Ranked]:
    """Each model's top `k` for each query."""
    retrieved = hits(path, corpus.ignore_identical_ids)
    out = []
    for qid, text in queries.items():
        scores = retrieved.get(qid, {})
        ids = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
        out.append(Ranked(qid, text, ids, [corpus.docs[d] for d in ids]))
    return out


def ndcg_at_10(path: Path, qrels: dict[str, dict[str, int]], ignore_identical_ids: bool = False) -> float:
    """mteb's own nDCG@10 of the prediction file against `qrels`."""
    from mteb._evaluators.retrieval_metrics import calculate_retrieval_scores

    return float(calculate_retrieval_scores(hits(path, ignore_identical_ids), qrels, [10]).ndcg["NDCG@10"])
