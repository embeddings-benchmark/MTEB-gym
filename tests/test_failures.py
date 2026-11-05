"""What a run does when judge calls fail: it never ranks on them silently, and a rerun repairs them."""

import json
import types

import pytest

from mteb_gym.cache import judge_pair_cached, judge_pool_cached
from mteb_gym.judge import PairwiseJudge, PointwiseJudge, Verdict
from mteb_gym.queries import Query, QueryGenerator
from mteb_gym.retrieval import Ranked
from mteb_gym.run import _cached_queries, _check_coverage, _check_judging, _judge_pointwise, _Judging, resolve_queries


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
    assert broken.calls == 4 and not any(g.parsed_ok for g in grades)  # each asked twice
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
    """Counting the ungraded document as 0 would hand q_failed to b. Without it, a wins every query that remains."""
    a = {f"q{i}": ["good"] for i in range(10)} | {"q_failed": ["d_ungraded"]}
    b = {f"q{i}": ["bad"] for i in range(10)} | {"q_failed": ["d_ok"]}
    grades = {"good": 5, "bad": 0, "d_ungraded": None, "d_ok": 5}
    corp = types.SimpleNamespace(name="task", docs={d: d for d in grades})
    client = Counting(grade_by_document(grades))
    ratings, diagnostics, _ = _judge_pointwise(
        client, "t", ranked({"a": a, "b": b}), _Judging(corp, tmp_path, "qs", 1, 2000, 0, 1)
    )
    assert {r.name: r.rating for r in ratings} == {"a": 100.0, "b": 0.0}
    assert diagnostics["queries_left_out"] == 1 and diagnostics["queries_rated"] == 10


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
    with pytest.raises(RuntimeError, match="failed|no usable"):
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


def test_only_the_current_pool_is_returned(tmp_path):
    """Grades saved for a model no longer on the roster are not returned."""
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
        Counting(grade_by_document(grades)), "t", lists, _Judging(corp, tmp_path, "qs", 1, 2000, 0, 1)
    )
    assert {r.name: r.rating for r in ratings}["b"] == pytest.approx(20.0)


def test_too_many_queries_left_out_is_an_error(tmp_path):
    """One failed grade loses a whole query, so a small rate of failed grades can lose most of the queries."""
    queries = {f"q{i}": [f"d{i}"] for i in range(20)}
    grades = {f"d{i}": (None if i < 3 else 5) for i in range(20)}
    corp = types.SimpleNamespace(name="task", docs={d: d for d in grades})
    with pytest.raises(RuntimeError, match="3 of 20"):
        _judge_pointwise(
            Counting(grade_by_document(grades)),
            "t",
            ranked({"m": queries}),
            _Judging(corp, tmp_path, "qs", 1, 2000, 0, 1),
        )


def test_a_failed_grade_is_asked_again_within_the_run(tmp_path):
    answers = iter(["no json", json.dumps({"grade": 3})])
    client = Counting(lambda m: next(answers))
    grades = judge_pool_cached(
        tmp_path, PointwiseJudge(client, "t", workers=1), ranked({"m": {"q": ["d"]}}), {"d": "d"}, "k"
    )
    assert client.calls == 2 and grades[0].parsed_ok and grades[0].grade == 3


def test_a_model_left_with_few_comparisons_is_an_error():
    ok = dict(raw=["A", "A"], parsed_ok=[True, True])
    bad = dict(raw=["tie", "tie"], parsed_ok=[False, False])
    good = [Verdict(f"q{i}", "q", "a", "b", 1.0, **ok) for i in range(4)] + [Verdict("q0", "q", "a", "c", 1.0, **ok)]
    _check_coverage(good)
    with pytest.raises(RuntimeError, match=r"\['c'\]"):
        _check_coverage([*good, *[Verdict(f"q{i}", "q", "a", "c", 0.5, **bad) for i in range(1, 4)]])


def test_identical_retrieval_everywhere_is_an_error():
    ratings = [types.SimpleNamespace(name="a")]
    with pytest.raises(RuntimeError, match="identical"):
        _check_judging({"identical_retrieval_rate": 1.0}, ratings, ["a"])


def test_queries_survive_a_filesystem_without_hard_links(tmp_path, monkeypatch):
    def no_links(src, dst):
        raise PermissionError("no hard links here")

    monkeypatch.setattr("os.link", no_links)
    gen = _generator()
    queries, _, _ = _cached_queries(tmp_path / "set.json", gen, {f"d{i}": "document" for i in range(9)})
    assert queries and json.loads((tmp_path / "set.json").read_text())["queries"] and not list(tmp_path.glob("*.tmp*"))
