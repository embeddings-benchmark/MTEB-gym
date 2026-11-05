"""Retrieval is mteb.evaluate on the gym task; the gym only reads mteb's prediction file."""

from __future__ import annotations

import gc
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .corpus import Corpus


@dataclass
class Ranked:
    """One model's top-k for one query."""

    qid: str
    query: str
    doc_ids: list[str]
    doc_texts: list[str]


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def predict(model_name: str, task, folder: Path, *, batch_size: int = 32) -> Path:
    """Run `model_name` on `task` through mteb, writing mteb's prediction file into
    `folder`; skipped when the file already exists. Dense, sparse and
    late-interaction models all go through the same call."""
    path = folder / f"{task.metadata.name}_predictions.json"
    if not path.exists():
        import mteb

        mteb.evaluate(
            mteb.get_model(model_name),
            task,
            prediction_folder=folder,
            cache=None,
            encode_kwargs={"batch_size": batch_size},
            show_progress_bar=False,
        )
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # free GPU memory between models
    return path


def hits(path: Path, ignore_identical_ids: bool = False) -> dict[str, dict[str, float]]:
    """What each model retrieved, from mteb's prediction file. A query that is itself a document is
    dropped from its own results where mteb's flag says so (ArguAna, FiQA), as mteb does when it scores."""
    scores = json.loads(path.read_text())["default"]["test"]
    if ignore_identical_ids:
        scores = {qid: {d: s for d, s in rels.items() if d != qid} for qid, rels in scores.items()}
    return scores


def top_k(path: Path, corpus: Corpus, queries: dict[str, str], k: int) -> list[Ranked]:
    """Each model's top `k` for each query."""
    retrieved = hits(path, corpus.ignore_identical_ids)
    out = []
    for qid, text in queries.items():
        scores = retrieved.get(qid, {})
        ids = sorted(scores, key=scores.get, reverse=True)[:k]
        out.append(Ranked(qid, text, ids, [corpus.docs[d] for d in ids]))
    return out


def ndcg_at_10(path: Path, qrels: dict[str, dict[str, int]], ignore_identical_ids: bool = False) -> float:
    """mteb's own nDCG@10 of the prediction file against `qrels`."""
    from mteb._evaluators.retrieval_metrics import calculate_retrieval_scores

    return float(calculate_retrieval_scores(hits(path, ignore_identical_ids), qrels, [10]).ndcg["NDCG@10"])
