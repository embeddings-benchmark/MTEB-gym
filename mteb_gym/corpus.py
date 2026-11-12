"""
A corpus is {doc_id: "title text"} plus the mteb task metadata that tells models
how to encode it (prompts, language). Loaded from an mteb retrieval task, or
from the documents' texts, numbered d0, d1, ...
Identity follows mteb: dataset path@revision (plus subset and split); only a
local corpus, which has no revision, is identified by a content hash.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mteb.abstasks.retrieval import AbsTaskRetrieval


@dataclass
class Corpus:
    name: str
    id: str  # mteb: <dataset path>@<revision>/<subset>/<split>; local: local:docs@<content hash>
    docs: dict[str, str]
    metadata: object  # mteb TaskMetadata
    queries: dict[str, str] | None = None  # the task's own queries
    qrels: dict[str, dict[str, int]] | None = None  # the task's own relevance labels
    ignore_identical_ids: bool = False  # mteb's flag: a query that is itself a document, as in ArguAna
    top_ranked: dict[str, list[str]] | None = None  # mteb's documents each of the task's own queries may retrieve

    @property
    def source(self) -> str:
        return "local" if self.id.startswith("local:") else "mteb"


def load(spec: AbsTaskRetrieval | Iterable[str]) -> Corpus:
    """`spec` is an mteb retrieval task, as mteb.get_task returns it, or the documents' texts."""
    if isinstance(spec, str | Path | Mapping):
        raise ValueError(
            f"a corpus is an mteb task or the documents' texts, not a {type(spec).__name__}: "
            "pass mteb.get_task(name), or the texts as a list"
        )
    if isinstance(spec, Iterable):
        return _load_local(spec)
    return _load_task(spec)


def _text(row: Mapping[str, str | None]) -> str:
    return ((row.get("title") or "") + " " + (row.get("text") or "")).strip()


def _load_task(task: AbsTaskRetrieval) -> Corpus:
    name = task.metadata.name
    task.load_data()
    task.convert_v1_dataset_format_to_v2(num_proc=None)  # as mteb.evaluate does: some tasks still load the v1 layout
    subsets = task.dataset
    if subsets is None:
        raise ValueError(f"{name} has no loaded dataset")
    filled = [name_ for name_, splits in subsets.items() if splits]
    if "en" in filled:  # a multilingual task runs on its English subset, as an English corpus
        filled = ["en"]
    if len(filled) > 1:
        raise ValueError(
            f"{name} has several subsets ({', '.join(sorted(filled))}); the gym runs on one corpus at a time"
        )
    subset = filled[0]
    split = "test" if "test" in subsets[subset] else next(iter(subsets[subset]))
    data = subsets[subset][split]
    docs = {row["id"]: _text(row) for row in data["corpus"]}

    meta = task.metadata
    dataset = meta.dataset or {}
    langs = meta.eval_langs[subset] if isinstance(meta.eval_langs, dict) else meta.eval_langs
    metadata = meta.model_copy(update={"eval_splits": ["test"], "eval_langs": list(langs)})  # one subset, one split
    queries = {row["id"]: row["text"] for row in data["queries"]} if "queries" in data else None
    qrels = data.get("relevant_docs")
    top_ranked = data.get("top_ranked")
    return Corpus(
        name=name,
        id=f"{dataset.get('path')}@{dataset.get('revision')}/{subset}/{split}",
        docs=docs,
        metadata=metadata,
        queries=queries,
        qrels={qid: dict(rels) for qid, rels in qrels.items()} if qrels is not None else None,
        ignore_identical_ids=task.ignore_identical_ids,
        top_ranked=dict(top_ranked) if top_ranked else None,
    )


def _load_local(texts: Iterable[str]) -> Corpus:
    from mteb.abstasks.task_metadata import TaskMetadata

    docs = {f"d{i}": text for i, text in enumerate(texts)}
    if not docs:
        raise ValueError("no documents given")
    if not all(isinstance(text, str) for text in docs.values()):
        raise ValueError("a document is a text: pass mteb.get_task(name) for an mteb task, or the documents' texts")
    h = hashlib.sha256()
    for did, text in docs.items():
        h.update(did.encode())
        h.update(b"\x00")
        h.update(text.encode())
        h.update(b"\x01")
    digest = h.hexdigest()[:12]
    # no benchmark task behind a local corpus: minimal metadata
    metadata = TaskMetadata(
        name="docs",
        description="Local corpus",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        dataset={"path": "docs", "revision": digest},
    )
    return Corpus(name="docs", id=f"local:docs@{digest}", docs=docs, metadata=metadata)
