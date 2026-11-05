"""A cached file is reused exactly when what produced it is the same."""

import pytest

from mteb_gym.cache import grades_key, model_id, sha, verdict_key
from mteb_gym.llm import MockLLM
from mteb_gym.queries import QueryGenerator
from mteb_gym.results import git_revision

BASE_VERDICT = dict(
    judge_id="j", system="s", top_k=10, doc_chars=2000, query_set="qs", a="a", rev_a="1", b="b", rev_b="2"
)
BASE_GRADES = dict(judge_id="j", system="s", instruction="i", doc_chars=2000, query_set="qs")


@pytest.mark.parametrize("field", list(BASE_VERDICT))
def test_every_part_of_a_verdict_key_counts(field):
    assert verdict_key(**BASE_VERDICT) != verdict_key(**{**BASE_VERDICT, field: "changed"})


@pytest.mark.parametrize("field", list(BASE_GRADES))
def test_every_part_of_a_grades_key_counts(field):
    assert grades_key(**BASE_GRADES) != grades_key(**{**BASE_GRADES, field: "changed"})


def test_keys_do_not_collide_across_part_boundaries():
    assert sha("a", "b") != sha("a|b")
    assert sha(["a", "b"]) != sha(["a|b"])


def test_settings_change_a_clients_identity_and_their_order_does_not():
    class Client:
        model = "m"
        max_tokens = None
        temperature = None
        extra_body = None

    def built(**kw):
        client = Client()
        for k, v in kw.items():
            setattr(client, k, v)
        return model_id(client)

    assert built() == "m"
    assert len({built(), built(temperature=1.0), built(max_tokens=5), built(extra_body={"a": 1})}) == 4
    assert built(extra_body={"a": 1, "b": 2}) == built(extra_body={"b": 2, "a": 1})


@pytest.mark.parametrize(
    "change",
    [
        {"task_instruction": "other"},
        {"n_queries": 3},
        {"seed": 1},
        {"filter_queries": False},
        {"docs_per_query": 2},
        {"overshoot": 2.0},
        {"min_chars": 9},
        {"max_chars": 9},
        {"min_score": 1},
        {"dedup": 0.5},
    ],
)
def test_every_setting_is_part_of_a_query_sets_identity(change):
    kw = dict(task_instruction="t", n_queries=2, workers=1)
    assert QueryGenerator(MockLLM(), **kw).params != QueryGenerator(MockLLM(), **{**kw, **change}).params


def test_an_installed_package_reports_no_commit_of_its_own(tmp_path, monkeypatch):
    """Installed in a project's venv, the package sits inside the project's git repository."""
    import subprocess

    from mteb_gym import results

    repo = tmp_path / "project"
    module = repo / "venv" / "site-packages" / "mteb_gym" / "results.py"
    module.parent.mkdir(parents=True)
    module.write_text("")
    subprocess.run(["git", "init", "--quiet", str(repo)], check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "--quiet", "--allow-empty", "-m", "x"],
        cwd=repo,
        check=True,
    )
    monkeypatch.setattr(results, "__file__", str(module))
    assert git_revision() is None


def test_a_checkout_reports_its_commit(tmp_path, monkeypatch):
    import subprocess

    from mteb_gym import results

    module = tmp_path / "mteb_gym" / "results.py"
    module.parent.mkdir()
    module.write_text("")
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "--quiet", "--allow-empty", "-m", "x"],
        cwd=tmp_path,
        check=True,
    )
    monkeypatch.setattr(results, "__file__", str(module))
    assert len(git_revision()) == 40
