"""End to end: a judge that can tell better documents from worse ones puts the better model first."""

import json

import numpy as np
import pytest

from mteb_gym import run
from mteb_gym.retrieval import Ranked
from tests.helpers import metas

GOOD, OK, BAD = "mteb/baseline-bm25s", "sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"
GRADE = {"great": 5.0, "fine": 2.0, "useless": 0.0}
WORD = {GOOD: "great", OK: "fine", BAD: "useless"}
QUERIES = ["first need", "second need", "third need", "fourth need"]


@pytest.fixture
def retrieved(monkeypatch):
    """Each model retrieves documents of a known quality, without running any model."""

    def top_k(path, corpus, texts, k):
        word = WORD[path]
        return [Ranked(q, t, [f"{word} {q}"], [f"{word} {q}"]) for q, t in texts.items()]

    monkeypatch.setattr("mteb_gym.retrieval.predict", lambda model, task, folder, **kw: model.name)
    monkeypatch.setattr("mteb_gym.retrieval.top_k", top_k)


def docs(tmp_path):
    path = tmp_path / "corpus.jsonl"
    rows = [{"id": f"{w} q{i}", "text": f"{w} document"} for w in GRADE for i in range(len(QUERIES))]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


class Decider:
    model = "decider"

    def decide(self, kind, state, question, options=None):
        document = state.split("Document: ")[1]
        expected = next(g for w, g in GRADE.items() if w in document)
        probabilities = np.zeros(6)
        probabilities[int(expected)] = 1.0
        return {str(i): float(p) for i, p in enumerate(probabilities)}


class Comparer:
    model = "comparer"

    def chat(self, messages):
        a, b = messages[-1]["content"].split("System B results:")
        best = max(GRADE, key=lambda w: GRADE[w] if w in messages[-1]["content"] else -1)
        return json.dumps({"winner": "A" if best in a else "B" if best in b else "tie"})


@pytest.mark.parametrize(
    ("judge", "judging"), [(Decider(), "pointwise"), (Comparer(), "pairwise")], ids=["pointwise", "pairwise"]
)
def test_the_better_model_ranks_first(tmp_path, retrieved, judge, judging):
    result = run(
        docs(tmp_path),
        metas([BAD, GOOD, OK]),
        judge,
        judging=judging,
        queries=QUERIES,
        task_instruction="t",
        output_folder=tmp_path / "o",
        cache_folder=tmp_path / "c",
        workers=1,
    )
    assert [r["model"] for r in result.record["ratings"]] == [GOOD, OK, BAD]
    ratings = [r["rating"] for r in result.record["ratings"]]
    assert ratings[0] > ratings[1] > ratings[2]


def test_pointwise_rating_matches_the_hand_computed_ndcg(tmp_path, retrieved):
    result = run(
        docs(tmp_path),
        metas([BAD, GOOD, OK]),
        Decider(),
        queries=QUERIES,
        task_instruction="t",
        output_folder=tmp_path / "o",
        cache_folder=tmp_path / "c",
        workers=1,
        top_k=1,
    )
    assert {r["model"]: r["rating"] for r in result.record["ratings"]} == {
        GOOD: pytest.approx(100.0),
        OK: pytest.approx(40.0),
        BAD: pytest.approx(0.0),
    }


def test_a_run_stops_if_failed_answers_leave_a_model_short(tmp_path, retrieved):
    class FailsOnOneModel(Comparer):
        def chat(self, messages):
            return "not json" if "useless" in messages[-1]["content"] else super().chat(messages)

    with pytest.raises(RuntimeError, match="most comparisons failed"):
        run(
            docs(tmp_path),
            metas([BAD, GOOD, OK]),
            FailsOnOneModel(),
            queries=QUERIES,
            judging="pairwise",
            task_instruction="t",
            output_folder=tmp_path / "o",
            cache_folder=tmp_path / "c",
            workers=1,
        )
    assert not list((tmp_path / "o").rglob("*.json"))


def test_a_run_stops_if_every_model_retrieved_the_same_lists(tmp_path, retrieved, monkeypatch):
    def same_list(path, corpus, texts, k):
        return [Ranked(q, t, [f"great {q}"], [f"great {q}"]) for q, t in texts.items()]

    monkeypatch.setattr("mteb_gym.retrieval.top_k", same_list)
    with pytest.raises(RuntimeError, match="identical"):
        run(
            docs(tmp_path),
            metas([BAD, GOOD, OK]),
            Comparer(),
            queries=QUERIES,
            judging="pairwise",
            task_instruction="t",
            output_folder=tmp_path / "o",
            cache_folder=tmp_path / "c",
            workers=1,
        )
    assert not list((tmp_path / "o").rglob("*.json"))
