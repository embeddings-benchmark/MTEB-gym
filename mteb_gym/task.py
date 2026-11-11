"""
The gym hands mteb a retrieval task and lets mteb run every model on it. The
task is the corpus plus the gym's queries; the qrels are each synthetic query's
seed documents (a diagnostic, not a label) or, for the task's own queries, its
own labels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from .corpus import Corpus
from .queries import Query

if TYPE_CHECKING:
    from mteb.abstasks.retrieval import AbsTaskRetrieval
    from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData


def labels(corpus: Corpus, queries: list[Query] | None, texts: dict[str, str]) -> dict[str, dict[str, int]] | None:
    """What a run can be scored against without a judge, on exactly the queries it judges: the dataset's own qrels for
    its own queries, each synthetic query's seed documents, nothing for queries you supplied."""
    if queries is None:
        if corpus.qrels is None:
            raise ValueError(f"{corpus.name} has no labels of its own")
        return {q: corpus.qrels[q] for q in texts}
    qrels = {q.qid: {d: 1 for d in q.seed_doc_ids} for q in queries if q.seed_doc_ids}
    return qrels or None


def build(corpus: Corpus, queries: list[Query] | None, texts: dict[str, str]) -> AbsTaskRetrieval:
    """An mteb AbsTaskRetrieval over `corpus` and the queries in `texts`. `queries=None` means they are the task's own,
    with its own qrels and, where the task has them, its documents per query (`top_ranked`), which mteb applies. A
    generated query or one you supply has no such list, so it searches the whole corpus."""
    from datasets import Dataset
    from mteb.abstasks.retrieval import AbsTaskRetrieval

    if queries is None:
        if not corpus.qrels:
            raise ValueError(f"{corpus.name} has no labels of its own")
        qrels = {q: corpus.qrels[q] for q in texts}
    else:
        # mteb drops a query without a labelled document; the gym uses no labels, so a query
        # without seed documents (your own queries) gets a placeholder that only satisfies mteb
        placeholder = next(iter(corpus.docs))
        qrels = {q.qid: {d: 1 for d in q.seed_doc_ids} or {placeholder: 1} for q in queries}

    allowed = None
    if queries is None and corpus.top_ranked:
        allowed = {q: corpus.top_ranked[q] for q in texts if q in corpus.top_ranked} or None

    ids = list(corpus.docs)
    data: dict[str, dict[str, RetrievalSplitData]] = {
        "default": {
            "test": {
                "corpus": Dataset.from_dict(
                    {"id": ids, "title": [""] * len(ids), "text": [corpus.docs[i] for i in ids]}
                ),
                "queries": Dataset.from_dict({"id": list(texts), "text": list(texts.values())}),
                "relevant_docs": qrels,
                "top_ranked": allowed,
            }
        }
    }

    def load_data(self: AbsTaskRetrieval, **kwargs: Any) -> None:
        self.dataset = data
        self.data_loaded = True

    task_type = cast(
        type[AbsTaskRetrieval],
        type("GymTask", (AbsTaskRetrieval,), {"metadata": corpus.metadata, "load_data": load_data}),
    )
    return task_type()
