"""
A corpus is {doc_id: "title text"} plus the mteb task metadata that tells models
how to encode it (prompts, language). Loaded from an mteb retrieval task,
or from a local directory of .txt/.md files or a .jsonl with id/text fields.
Identity follows mteb: dataset path@revision (plus subset and split); only a
local corpus, which has no revision, is identified by a content hash.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mteb.abstasks.retrieval import AbsTaskRetrieval

logger = logging.getLogger(__name__)


@dataclass
class Corpus:
    name: str
    id: str  # mteb: <dataset path>@<revision>/<subset>/<split>; local: local:<name>@<content hash>
    docs: dict[str, str]
    metadata: object  # mteb TaskMetadata
    queries: dict[str, str] | None = None  # the task's own queries
    qrels: dict[str, dict[str, int]] | None = None  # the task's own relevance labels
    ignore_identical_ids: bool = False  # mteb's flag: a query that is itself a document, as in ArguAna
    top_ranked: dict[str, list[str]] | None = None  # mteb's documents each of the task's own queries may retrieve

    @property
    def source(self) -> str:
        return "local" if self.id.startswith("local:") else "mteb"


def load(spec: AbsTaskRetrieval | str | Path) -> Corpus:
    """`spec` is an mteb retrieval task, as mteb.get_task returns it, or a path to a directory / .jsonl of documents."""
    if not isinstance(spec, str | Path):
        return _load_task(spec)
    path = Path(spec)
    if not path.exists():
        raise ValueError(
            f"no documents at {path}: a corpus is a folder or .jsonl of documents, or an mteb task, "
            f"as mteb.get_task({str(spec)!r}) returns it"
        )
    logger.info("loading the local corpus at %s", path)
    return _load_local(path)


def _text(row) -> str:
    return ((row.get("title") or "") + " " + (row.get("text") or "")).strip()


def _load_task(task: AbsTaskRetrieval) -> Corpus:
    name = task.metadata.name
    task.load_data()
    task.convert_v1_dataset_format_to_v2(num_proc=None)  # as mteb.evaluate does: some tasks still load the v1 layout
    subsets = task.dataset
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
    return Corpus(
        name=name,
        id=f"{dataset.get('path')}@{dataset.get('revision')}/{subset}/{split}",
        docs=docs,
        metadata=metadata,
        queries=queries,
        qrels=data.get("relevant_docs"),
        ignore_identical_ids=task.ignore_identical_ids,
        top_ranked=dict(data["top_ranked"]) if data.get("top_ranked") else None,
    )


def _load_local(root: Path) -> Corpus:
    from mteb.abstasks.task_metadata import TaskMetadata

    if root.is_file() and root.suffix == ".jsonl":
        rows = (json.loads(line) for line in root.read_text().split("\n") if line.strip())
        docs = {str(r["id"]): str(r.get("text") or "") for r in rows}
    else:
        files = sorted(p for p in root.rglob("*") if p.suffix in (".txt", ".md"))
        docs = {str(p.relative_to(root)): p.read_text(errors="ignore") for p in files}
    if not docs:
        raise ValueError(f"no documents found in {root}")
    h = hashlib.sha256()
    for did, text in docs.items():
        h.update(did.encode())
        h.update(b"\x00")
        h.update(text.encode())
        h.update(b"\x01")
    digest = h.hexdigest()[:12]
    # no benchmark task behind a local corpus: minimal metadata
    metadata = TaskMetadata(
        name=root.stem,
        description=f"Local corpus {root.name}",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        dataset={"path": root.name, "revision": digest},
    )
    return Corpus(name=root.stem, id=f"local:{root.stem}@{digest}", docs=docs, metadata=metadata)
