import tempfile
import types
from pathlib import Path

from mteb_gym import Result, load_results, results
from mteb_gym.judge import Verdict
from mteb_gym.rank import rate


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
    exp = {"arm": "synthetic", "judge_model": "org/judge", "generator_model": "org/gen", "seed": 0, "n_queries": 3}
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


def test_git_revision_is_only_read_from_a_checkout(tmp_path, monkeypatch):
    """An installed package must not report the commit of whatever repository it sits in."""
    module = tmp_path / "site-packages" / "mteb_gym" / "results.py"
    module.parent.mkdir(parents=True)
    module.write_text("")
    monkeypatch.setattr(results, "__file__", str(module))
    assert results.git_revision() is None
