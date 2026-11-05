import json
import math
import tempfile
import types
from pathlib import Path

import pytest

from mteb_gym import Result, load_results, results
from mteb_gym.judge import Grade, Verdict
from mteb_gym.rank import rate
from mteb_gym.run import grade_diagnostics


def test_record():
    assert results.config_hash({"a": 1, "b": [2, 3]}) == results.config_hash({"b": [2, 3], "a": 1})
    verdicts = [
        Verdict("q0", "q", "a", "b", 1.0, raw=["A", "B"], parsed_ok=[True, True]),
        Verdict("q1", "q", "a", "b", 0.5, raw=["identical"]),
        Verdict("q2", "q", "a", "b", 0.5, raw=["tie", "tie"], parsed_ok=[False, True]),
    ]
    d = results.verdict_diagnostics(verdicts)
    assert d == {
        "judge_calls": 4,
        "n_comparisons": 3,
        "commit_rate": 1 / 3,
        "tie_rate": 2 / 3,
        "a_first_rate": 0.5,
        "parse_failure_rate": 0.25,
        "identical_retrieval_rate": 1 / 3,
    }
    corpus = types.SimpleNamespace(
        name="demo",
        id="local:demo@abc",
        source="local",
        metadata=types.SimpleNamespace(dataset={"path": "x", "revision": "y"}),
    )
    exp = {
        "query_source": "synthetic",
        "judge_model": "org/judge",
        "generator_model": "org/gen",
        "seed": 0,
        "n_queries": 3,
    }
    exp["config_hash"] = results.config_hash(exp)
    rec = results.build_record(corpus, exp, rate(verdicts, bootstrap=0), verdicts, 1.0, {"a": "r1", "b": None})
    assert rec["source"] == "local" and rec["diagnostics"]["tie_rate"] == 2 / 3
    assert (
        results.record_path(Path("out"), "demo", exp)
        == Path("out") / "demo" / f"demo__judge__gen__q3-s0-{exp['config_hash']}.json"
    )
    with tempfile.TemporaryDirectory() as tmp:
        r = Result(rec, Path(tmp) / "demo" / "demo__judge__original-queries.json")
        r.to_disk()
        again = Result.from_disk(r.path)
        assert again.record == rec and "demo" not in again.leaderboard and "a" in again.leaderboard
        df = load_results(tmp).to_dataframe()
        assert list(df["model"]) == [x["model"] for x in rec["ratings"]] and set(df["task"]) == {"demo"}
        # a results folder or results repository: <task>/<task>__....json, next to nothing else
        repo = Path(tmp) / "repo" / "demo"
        repo.mkdir(parents=True)
        (repo / "demo__judge__q2-s0-abc.json").write_text(r.path.read_text())
        assert len(load_results(Path(tmp) / "repo").results) == 1


def test_position_bias_is_the_share_of_decisive_answers_for_the_first_shown():
    verdicts = [
        Verdict("q0", "q", "a", "b", 0.5, raw=["A", "A"], parsed_ok=[True, True]),
        Verdict("q1", "q", "a", "b", 0.75, raw=["A", "B"], parsed_ok=[True, True]),
    ]
    assert results.verdict_diagnostics(verdicts)["a_first_rate"] == pytest.approx(0.75)


def test_grade_diagnostics_report_failures():
    grades = [
        Grade("q0", "d0", 4.0),
        Grade("q0", "d1", 2.0),
        Grade("q1", "d2", 0.0),
        Grade("q1", "d3", 0.0, parsed_ok=False),
    ]
    out = grade_diagnostics(grades)
    assert out["parse_failure_rate"] == 0.25 and out["grade_mean"] == pytest.approx(2.0) and out["pool_size_mean"] == 2


def test_only_record_files_in_the_results_layout_are_loaded(tmp_path):
    rec = {"task_name": "t", "ratings": []}
    (tmp_path / "t").mkdir()
    (tmp_path / "t" / "t__j__q.json").write_text(json.dumps(rec))
    (tmp_path / "t" / "notes.json").write_text("not a record")
    assert len(load_results(tmp_path).results) == 1


def test_a_query_that_is_itself_a_document_is_dropped_only_with_the_flag(tmp_path):
    from mteb_gym import retrieval

    path = tmp_path / "t_predictions.json"
    path.write_text(json.dumps({"default": {"test": {"q0": {"q0": 2.0, "d1": 1.0}}}}))
    assert retrieval.ndcg_at_10(path, {"q0": {"d1": 1}}, True) == pytest.approx(1.0)
    assert retrieval.ndcg_at_10(path, {"q0": {"d1": 1}}, False) == pytest.approx(1 / math.log2(3))


def test_predictions_on_disk_are_not_computed_again(tmp_path, monkeypatch):
    import mteb

    from mteb_gym import retrieval

    (tmp_path / "t_predictions.json").write_text("{}")
    monkeypatch.setattr(mteb, "evaluate", lambda *a, **k: pytest.fail("should not run"))
    task = types.SimpleNamespace(metadata=types.SimpleNamespace(name="t"))
    assert retrieval.predict("m", task, tmp_path) == tmp_path / "t_predictions.json"
