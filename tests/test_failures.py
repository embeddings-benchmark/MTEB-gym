"""What a run does when judge calls fail: it never ranks on them silently, and a rerun repairs them."""

import json
import types

import pytest

from mteb_gym.cache import judge_pair_cached, judge_pool_cached
from mteb_gym.judge import PairwiseJudge, PointwiseJudge
from mteb_gym.queries import Query, QueryGenerator
from mteb_gym.retrieval import Ranked
from mteb_gym.run import _cached_queries, _check_judging, _judge_pointwise, resolve_queries


class Counting:
    """A chat judge with a fixed answer, counting its calls."""

    model = "counting"

    def __init__(self, answer):
        self.answer, self.calls = answer, 0

    def chat(self, messages):
        self.calls += 1
        return self.answer(messages) if callable(self.answer) else self.answer


def grade_by_document(grades):
    """Answers with the grade given for the document named in the prompt; garbage when it has none."""

    def answer(messages):
        for doc, grade in grades.items():
            if doc in messages[-1]["content"]:
                return json.dumps({"grade": grade}) if grade is not None else "no json here"
        raise AssertionError(messages[-1]["content"])

    return answer


def ranked(model_docs):
    """{model: {qid: [doc ids]}} as Ranked lists; a document's text is its id."""
    return {
        m: [Ranked(q, f"query {q}", ids, list(ids)) for q, ids in queries.items()] for m, queries in model_docs.items()
    }


def test_failed_grades_are_asked_again_on_a_rerun(tmp_path):
    lists = ranked({"m": {"q0": ["d0", "d1"]}})
    docs = {"d0": "d0", "d1": "d1"}
    broken = Counting("no json here")
    grades = judge_pool_cached(tmp_path, PointwiseJudge(broken, "t", workers=1), lists, docs, "k")
    assert broken.calls == 2 and not any(g.parsed_ok for g in grades)
    healthy = Counting(json.dumps({"grade": 4}))
    grades = judge_pool_cached(tmp_path, PointwiseJudge(healthy, "t", workers=1), lists, docs, "k")
    assert healthy.calls == 2 and all(g.parsed_ok and g.grade == 4 for g in grades)
    healthy.calls = 0
    judge_pool_cached(tmp_path, PointwiseJudge(healthy, "t", workers=1), lists, docs, "k")
    assert healthy.calls == 0  # a usable answer is not asked again


def test_failed_verdicts_are_asked_again_on_a_rerun(tmp_path):
    a = ranked({"a": {"q0": ["d0"]}})["a"]
    b = ranked({"b": {"q0": ["d1"]}})["b"]
    broken = Counting("no json here")
    v = judge_pair_cached(tmp_path, PairwiseJudge(broken, "t"), "a", "b", a, b, "k")
    assert v[0].parsed_ok == [False, False]
    healthy = Counting(json.dumps({"winner": "A"}))
    v = judge_pair_cached(tmp_path, PairwiseJudge(healthy, "t"), "a", "b", a, b, "k")
    assert healthy.calls == 2 and all(v[0].parsed_ok)


def test_a_query_with_a_failed_grade_does_not_count_against_any_model(tmp_path):
    """With d_ungraded counted as 0, b would beat a on q1. Without q1, the order comes from q0 alone."""
    lists = ranked({"a": {"q0": ["good"], "q1": ["d_ungraded"]}, "b": {"q0": ["bad"], "q1": ["d_ok"]}})
    grades = {"good": 5, "bad": 0, "d_ungraded": None, "d_ok": 5}
    corp = types.SimpleNamespace(name="task", docs={d: d for d in grades})
    client = Counting(grade_by_document(grades))
    ratings, diagnostics, _ = _judge_pointwise(client, "t", corp, lists, tmp_path, "qs", 1, 2000, 0, 1)
    by_model = {r.name: r.rating for r in ratings}
    assert by_model == {"a": 100.0, "b": 0.0}
    assert diagnostics["queries_left_out"] == 1


@pytest.mark.parametrize(
    "diagnostics, rated",
    [
        ({"parse_failure_rate": 1.0}, ["a", "b"]),  # every judge answer failed
        ({"parse_failure_rate": 0.6}, ["a", "b"]),  # most of them failed
        ({"parse_failure_rate": 0.0}, ["a"]),  # a model lost all its comparisons to failures
        ({"parse_failure_rate": 0.0}, []),
    ],
)
def test_a_run_the_judge_failed_is_an_error_not_a_record(diagnostics, rated):
    ratings = [types.SimpleNamespace(name=m) for m in rated]
    with pytest.raises(RuntimeError):
        _check_judging(diagnostics, ratings, ["a", "b"])


def test_a_healthy_run_passes_the_check():
    ratings = [types.SimpleNamespace(name=m) for m in ("a", "b")]
    _check_judging({"parse_failure_rate": 0.1}, ratings, ["a", "b"])


class Scripted:
    """A generator whose queries carry a label, so two runs can write different queries."""

    model = "scripted"

    def __init__(self, label):
        self.label, self.calls = label, 0

    def chat(self, messages):
        self.calls += 1
        return json.dumps({"query": f"{self.label} query number {self.calls}"})


def _generator(label="first"):
    return QueryGenerator(Scripted(label), task_instruction="t", n_queries=3, filter_queries=False, workers=1)


def test_the_first_writer_of_a_query_set_wins(tmp_path):
    """Two processes generating the same set use one set of queries, not one each."""
    path = tmp_path / "set.json"
    winner = [Query("q0", "the winner's query", [])]
    gen = _generator()
    real_run = gen.run

    def run_while_another_process_finishes_first(docs):
        out = real_run(docs)
        path.write_text(json.dumps({"generator": None, "n_generated": 1, "queries": [vars(winner[0])]}))
        return out

    gen.run = run_while_another_process_finishes_first
    queries, _, _ = _cached_queries(path, gen, {f"d{i}": f"document number {i} about things" for i in range(9)})
    assert [q.text for q in queries] == ["the winner's query"]
    assert not list(tmp_path.glob("*.tmp*"))


def test_regenerated_queries_never_reuse_older_predictions(tmp_path):
    """A query set is identified by its queries too: if its file is lost and generation gives other queries, what was
    saved for the old ones must not be read back for the new ones."""
    corp = types.SimpleNamespace(
        name="task", id="task@1", docs={f"d{i}": f"document number {i} about things" for i in range(9)}, queries=None
    )
    first = resolve_queries(corp, tmp_path, "synthetic", _generator("first"), 3, 0)
    for f in (tmp_path / "queries").rglob("*.json"):
        f.unlink()
    second = resolve_queries(corp, tmp_path, "synthetic", _generator("second"), 3, 0)
    assert first.texts != second.texts and first.id != second.id
    assert resolve_queries(corp, tmp_path, "synthetic", _generator("second"), 3, 0).id == second.id


def test_only_the_current_pool_is_returned_and_rated(tmp_path):
    """Grades saved for a model no longer on the roster do not change the ideal ranking of the rest."""
    docs = {"d0": "d0", "d1": "d1"}
    client = Counting(json.dumps({"grade": 5}))
    judge = PointwiseJudge(client, "t", workers=1)
    judge_pool_cached(tmp_path, judge, ranked({"gone": {"q0": ["d1"]}, "m": {"q0": ["d0"]}}), docs, "k")
    grades = judge_pool_cached(tmp_path, judge, ranked({"m": {"q0": ["d0"]}}), docs, "k")
    assert [g.doc_id for g in grades] == ["d0"]


def test_only_the_requested_queries_of_a_pair_are_returned(tmp_path):
    a = ranked({"a": {"q0": ["d0"], "q1": ["d0"]}})["a"]
    b = ranked({"b": {"q0": ["d1"], "q1": ["d1"]}})["b"]
    judge = PairwiseJudge(Counting(json.dumps({"winner": "A"})), "t")
    judge_pair_cached(tmp_path, judge, "a", "b", a, b, "k")
    assert [v.qid for v in judge_pair_cached(tmp_path, judge, "a", "b", a[:1], b[:1], "k")] == ["q0"]


def test_pointwise_rating_uses_the_runs_top_k(tmp_path):
    lists = ranked({"a": {"q0": ["good"]}, "b": {"q0": ["weak"]}})
    grades = {"good": 5, "weak": 1}
    corp = types.SimpleNamespace(name="task", docs={d: d for d in grades})
    ratings, _, _ = _judge_pointwise(
        Counting(grade_by_document(grades)), "t", corp, lists, tmp_path, "qs", 1, 2000, 0, 1
    )
    assert {r.name: r.rating for r in ratings}["b"] == pytest.approx(20.0)
