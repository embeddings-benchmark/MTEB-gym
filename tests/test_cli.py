"""The command line passes what it is given."""

import types

import pytest

from mteb_gym import cli
from mteb_gym.llm import DecisionModel, MockLLM


@pytest.fixture
def run_call(monkeypatch, tmp_path):
    seen = {}

    def fake_run(corpus, models, judge, **kw):
        seen.update(corpus=corpus, models=models, judge=judge, **kw)
        return types.SimpleNamespace(leaderboard="board", record={"diagnostics": {}}, path=tmp_path / "r.json")

    monkeypatch.setattr(cli, "run", fake_run)
    return seen


def test_run_passes_its_arguments(run_call, capsys):
    cli.main(
        "run --corpus docs --models a b --judge mock --generator mock --task-instruction t --n-queries 7 "
        "--queries original --top-k 3 --doc-chars 9 --pairs-per-query 2 --judging pairwise --seed 5 "
        "--batch-size 4 --workers 6 --embedding-cache vectors --max-seq-length 8192".split()
    )
    assert run_call["corpus"] == "docs" and run_call["models"] == ["a", "b"]
    assert isinstance(run_call["judge"], MockLLM) and isinstance(run_call["generator"], MockLLM)
    expected = dict(
        task_instruction="t",
        n_queries=7,
        queries="original",
        top_k=3,
        doc_chars=9,
        pairs_per_query=2,
        judging="pairwise",
        seed=5,
        filter_queries=False,
        batch_size=4,
        workers=6,
        embedding_cache="vectors",
        max_seq_length=8192,
    )
    assert {k: run_call[k] for k in expected} == expected
    assert "board" in capsys.readouterr().out


def test_a_decision_judge_needs_a_url(run_call, capsys):
    with pytest.raises(SystemExit):
        cli.main("run --corpus d --models a --judge j --judge-decision".split())
    cli.main("run --corpus d --models a --judge j --judge-decision --judge-url http://x".split())
    assert isinstance(run_call["judge"], DecisionModel) and run_call["judge"].base_url == "http://x"


def test_predict_is_one_model(monkeypatch, capsys):
    seen = {}
    monkeypatch.setattr(
        cli, "predict", lambda corpus, model, **kw: seen.update(corpus=corpus, model=model, **kw) or "p"
    )
    cli.main("predict --corpus d --model a --queries original".split())
    assert seen["model"] == "a" and seen["queries"] == "original" and "predictions -> p" in capsys.readouterr().out


def test_each_endpoint_can_use_its_own_key_variable(run_call, monkeypatch):
    monkeypatch.setenv("JUDGE_KEY", "judge-secret")
    monkeypatch.setenv("GEN_KEY", "generator-secret")
    cli.main(
        "run --corpus d --models a --judge j --judge-url http://j --judge-decision --judge-key-env JUDGE_KEY "
        "--generator g --generator-url http://g/v1 --generator-key-env GEN_KEY".split()
    )
    assert run_call["judge"]._http.headers["authorization"] == "Bearer judge-secret"
    assert run_call["generator"].client.api_key == "generator-secret"
