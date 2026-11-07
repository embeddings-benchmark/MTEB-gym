"""The command line passes what it is given."""

import types

import pytest

from mteb_gym import cli
from mteb_gym.llm import MockLLM
from mteb_gym.models.jev import JevServer as DecisionModel


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


def test_a_judge_whose_repo_is_a_decision_model_is_one(run_call, monkeypatch):
    monkeypatch.setattr("mteb_gym.llm.repo_files", lambda name: {"judge_config.json"})
    cli.main("run --corpus d --models a --judge j --judge-url http://x".split())
    assert isinstance(run_call["judge"], DecisionModel) and run_call["judge"].base_url == "http://x"


def test_each_endpoint_can_use_its_own_key_variable(run_call, monkeypatch):
    monkeypatch.setenv("JUDGE_KEY", "judge-secret")
    monkeypatch.setenv("GEN_KEY", "generator-secret")
    monkeypatch.setattr("mteb_gym.llm.repo_files", lambda name: {"judge_config.json"} if name == "j" else None)
    cli.main(
        "run --corpus d --models a --judge j --judge-url http://j --judge-key-env JUDGE_KEY "
        "--generator g --generator-url http://g/v1 --generator-key-env GEN_KEY".split()
    )
    assert run_call["judge"]._http.headers["authorization"] == "Bearer judge-secret"
    assert run_call["generator"].client.api_key == "generator-secret"
