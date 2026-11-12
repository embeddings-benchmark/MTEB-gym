"""A cached file is reused exactly when what produced it is the same."""

import json
import types

import pytest

from mteb_gym import corpus as corpus_mod
from mteb_gym import task as task_mod
from mteb_gym.cache import grades_key, model_id, sha, verdict_key
from mteb_gym.queries import Query, QueryGenerator
from mteb_gym.results import git_revision
from mteb_gym.run import own_queries, resolve_queries, sample_queries

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
        {"filter_queries": True},
        {"docs_per_query": 2},
        {"overshoot": 2.0},
        {"min_chars": 9},
        {"max_chars": 90},
        {"min_score": 1},
        {"dedup": 0.5},
    ],
)
def test_every_setting_is_part_of_a_query_sets_id(tmp_path, change):
    assert _set_id(tmp_path, **change) != _set_id(tmp_path)


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


def _corpus(**kw):
    return types.SimpleNamespace(
        **{"name": "t", "id": "t@1", "docs": DOCS, "top_ranked": None, "queries": {}, "qrels": {}, **kw}
    )


DOCS = {f"d{i}": f"document {i}" for i in range(9)}


class Named:
    def __init__(self, model):
        self.model = model

    def chat(self, messages):
        return json.dumps({"query": "the same query every time"})


def _set_id(tmp, corp=None, model="m", **kw):
    gen = QueryGenerator(
        Named(model), **{"task_instruction": "t", "n_queries": 1, "filter_queries": False, "workers": 1, **kw}
    )
    return resolve_queries(corp or _corpus(), tmp, "synthetic", gen, 1, 0).id


def test_a_generated_sets_id_depends_on_how_it_was_made(tmp_path):
    base = _set_id(tmp_path)
    assert _set_id(tmp_path) == base
    assert _set_id(tmp_path, seed=1) != base
    assert _set_id(tmp_path, model="other") != base
    assert _set_id(tmp_path, _corpus(id="other@1")) != base


def test_a_changed_prompt_is_a_new_query_set(tmp_path, monkeypatch):
    base = _set_id(tmp_path)
    monkeypatch.setattr("mteb_gym.queries._FILTER_BODY", "another prompt")
    filter_changed = _set_id(tmp_path)
    assert len({base, filter_changed, _set_id(tmp_path, doc_chars=1)}) == 3


def test_original_and_own_sets_are_identified_by_their_queries(tmp_path):
    corp = _corpus(queries={f"q{i}": f"query {i}" for i in range(6)}, qrels={f"q{i}": {"d0": 1} for i in range(6)})
    gen = types.SimpleNamespace()
    assert (
        resolve_queries(corp, tmp_path, "original", gen, 2, 0).id
        != resolve_queries(corp, tmp_path, "original", gen, 2, 1).id
    )

    def own(texts):
        return resolve_queries(corp, tmp_path, texts, gen, 2, 0).id

    assert own(["a"]) != own(["b"]) and own(["a"]) == own(["a"])


def test_sampling_depends_on_the_seed_and_repeats():
    queries = {f"q{i}": "x" for i in range(50)}
    assert sample_queries(queries, 5, 0) == sample_queries(queries, 5, 0)
    assert sample_queries(queries, 5, 0) != sample_queries(queries, 5, 1)
    assert len(sample_queries(queries, 5, 0)) == 5 and sample_queries(queries, 99, 0) == queries


def test_own_queries_keep_their_ids(tmp_path):
    path = tmp_path / "queries.jsonl"
    path.write_text('{"id": "mine", "text": "a query"}\n')
    assert [(q.qid, q.text) for q in own_queries(path)] == [("mine", "a query")]


def test_a_local_corpus_is_identified_by_its_ids_and_text():
    base = corpus_mod.load(["one", "two"])
    assert base.docs == {"d0": "one", "d1": "two"}
    assert corpus_mod.load(["one", "three"]).id != base.id
    assert corpus_mod.load(["two", "one"]).id != base.id


def test_a_tasks_documents_include_their_titles():
    assert corpus_mod._text({"title": "Title", "text": "body"}) == "Title body"
    assert corpus_mod._text({"title": None, "text": "body"}) == "body"


def test_seed_labels_keep_every_seed_document():
    queries = [Query("q0", "a query", ["d0", "d1"])]
    corp = types.SimpleNamespace(qrels=None)
    assert task_mod.labels(corp, queries, {"q0": "a query"}) == {"q0": {"d0": 1, "d1": 1}}
    assert task_mod.labels(types.SimpleNamespace(qrels={"q0": {"d": 1}, "q1": {"d": 1}}), None, {"q0": "a"}) == {
        "q0": {"d": 1}
    }


def test_a_multilingual_task_runs_on_its_english_subset():
    from datasets import Dataset

    from mteb_gym import corpus as corpus_mod

    split = {
        "corpus": Dataset.from_dict({"id": ["d1", "d2"], "title": ["", ""], "text": ["one", "two"]}),
        "queries": Dataset.from_dict({"id": ["q1", "q2"], "text": ["first", "second"]}),
        "relevant_docs": {"q1": {"d1": 1}, "q2": {"d2": 1}},
        "top_ranked": {"q1": ["d1"], "q2": ["d2"]},
    }

    class Meta:
        name = "t"
        dataset = {"path": "p", "revision": "r"}
        eval_langs = {"en": ["eng-Latn"], "es": ["spa-Latn"]}
        prompt = None

        def model_copy(self, update):
            return self

    class Task:
        metadata = Meta()
        ignore_identical_ids = False
        dataset = {"en": {"all": split}, "es": {"all": {**split, "corpus": None}}}

        def load_data(self):
            pass

        def convert_v1_dataset_format_to_v2(self, num_proc=None):
            pass

    corpus = corpus_mod.load(Task())
    assert "/en/all" in corpus.id and set(corpus.docs) == {"d1", "d2"}
