import json
import math
import subprocess
import tempfile
import types
from pathlib import Path

import pytest

from mteb_gym import Result, load_results, results
from mteb_gym.judge import Grade, Verdict
from mteb_gym.rank import rate
from mteb_gym.results import grade_diagnostics


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
        "n_shortened": 0,
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
    rec = results.build_record(
        corpus, exp, rate(verdicts, bootstrap=0), results.verdict_diagnostics(verdicts), 1.0, {"a": "r1", "b": None}
    )
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
    assert retrieval.predict(_meta(None), task, tmp_path) == tmp_path / "t_predictions.json"


class _Task:
    metadata = types.SimpleNamespace(name="t")


def _meta(model, name="m", model_type=None):
    """A ModelMeta that loads `model`."""
    return types.SimpleNamespace(
        name=name, model_name_with_experiment=name, model_type=model_type, load_model=lambda: model
    )


def _patched(monkeypatch):
    """mteb with a run that only writes the prediction file. Returns what mteb was asked to run."""
    import mteb

    ran = []

    def evaluate(model, task, *, prediction_folder, **kw):
        ran.append(model)
        (Path(prediction_folder) / "t_predictions.json").write_text("{}")

    monkeypatch.setattr(mteb, "evaluate", evaluate)
    return ran


def _encoder(max_seq_length=32768):
    return types.SimpleNamespace(model=types.SimpleNamespace(max_seq_length=max_seq_length), encode=lambda *a: None)


def test_by_default_a_model_runs_as_mteb_loads_it(tmp_path, monkeypatch):
    from mteb_gym import retrieval

    model = _encoder()
    ran = _patched(monkeypatch)
    retrieval.predict(_meta(model), _Task, tmp_path / "p")
    assert ran == [model] and model.model.max_seq_length == 32768


@pytest.mark.parametrize(("own", "cap", "after"), [(32768, 8192, 8192), (512, 8192, 512), (8192, 8192, 8192)])
def test_the_cap_only_lowers_a_length(tmp_path, monkeypatch, own, cap, after):
    from mteb_gym import retrieval

    model = _encoder(own)
    _patched(monkeypatch)
    retrieval.predict(_meta(model), _Task, tmp_path / "p", max_seq_length=cap)
    assert model.model.max_seq_length == after


def test_the_cap_leaves_a_static_model_and_one_without_a_length(tmp_path, monkeypatch):
    from mteb_gym import retrieval

    class StaticEmbedding:
        max_seq_length = 1000000

    static = types.SimpleNamespace(model=StaticEmbedding(), encode=lambda *a: None)
    _patched(monkeypatch)
    retrieval.predict(_meta(static), _Task, tmp_path / "static", max_seq_length=8192)
    assert static.model.max_seq_length == 1000000
    lexical = types.SimpleNamespace(encode=lambda *a: None)
    retrieval.predict(_meta(lexical), _Task, tmp_path / "lexical", max_seq_length=8192)  # nothing to cap: it runs


def test_the_embedding_cache_wraps_a_model_at_its_own_folder(tmp_path, monkeypatch):
    from mteb.models.cache_wrappers import CachedEmbeddingWrapper

    from mteb_gym import retrieval

    ran = _patched(monkeypatch)
    retrieval.predict(_meta(_encoder(), "org/my model"), _Task, tmp_path / "p", embedding_cache=tmp_path / "vectors")
    assert isinstance(ran[0], CachedEmbeddingWrapper) and ran[0].cache_path == tmp_path / "vectors" / "org_my_model"


def test_models_without_one_vector_per_text_skip_the_embedding_cache(tmp_path, monkeypatch):
    from mteb_gym import retrieval

    class StaticEmbedding:
        pass

    def uncached(name, model, meta_type=None):
        ran = _patched(monkeypatch)
        retrieval.predict(_meta(model, name, meta_type), _Task, tmp_path / name, embedding_cache=tmp_path / "vectors")
        return ran == [model]

    assert uncached("mteb/baseline-bm25s", _encoder())
    assert uncached("static", types.SimpleNamespace(model=StaticEmbedding(), encode=lambda *a: None))
    assert uncached("colbert", _encoder(), ["late-interaction"])
    assert uncached("no-encode", types.SimpleNamespace())
    assert not (tmp_path / "vectors").exists()


def test_vectors_of_one_cap_are_kept_apart_from_another(tmp_path, monkeypatch):
    from mteb_gym import retrieval

    folder = {}
    for cap in (None, 8192, 512):
        ran = _patched(monkeypatch)
        retrieval.predict(
            _meta(_encoder()), _Task, tmp_path / str(cap), embedding_cache=tmp_path / "vectors", max_seq_length=cap
        )
        folder[cap] = ran[0].cache_path
    assert folder == {
        None: tmp_path / "vectors" / "m",
        8192: tmp_path / "vectors" / "m-max8192",
        512: tmp_path / "vectors" / "m-max512",
    }


def test_the_cache_reopens_its_own_vectors(tmp_path):
    import numpy as np
    from mteb.models.cache_wrappers.cache_backends.numpy_cache import NumpyCache

    from mteb_gym import retrieval

    model = types.SimpleNamespace(encode=lambda *a, **k: None)
    backend = retrieval._cached(model, tmp_path).cache_backend
    assert issubclass(backend, NumpyCache)
    first = backend(tmp_path / "c")
    first.add([{"text": "one"}, {"text": "two"}], np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32"))
    first.save()
    first.close()
    second = backend(tmp_path / "c")
    second.load()
    assert second.get_vector({"text": "two"}).tolist() == [3.0, 4.0]


def _commit_record(work, task, name):
    folder = work / "results" / task
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{task}__judge__{name}.json").write_text(json.dumps({"task_name": task, "ratings": []}))
    git = ["git", "-C", str(work), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", name], check=True)


def test_without_a_path_the_results_repository_is_fetched_into_the_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("MTEB_GYM_CACHE", str(tmp_path / "cache"))
    work, remote = tmp_path / "work", str(tmp_path / "remote.git")
    subprocess.run(["git", "init", "-q", "-b", "main", str(work)], check=True)
    _commit_record(work, "t", "one")
    subprocess.run(["git", "clone", "-q", "--bare", str(work), remote], check=True)
    assert len(load_results(results_repo=remote).results) == 1  # first call: cloned
    _commit_record(work, "t", "two")
    subprocess.run(["git", "-C", str(work), "push", "-q", remote, "main"], check=True)
    assert len(load_results(results_repo=remote, download_latest=False).results) == 1  # the clone as it is
    assert len(load_results(results_repo=remote).results) == 2  # moved to the latest commit
    subprocess.run(["git", "-C", str(work), "branch", "old", "HEAD~1"], check=True)
    subprocess.run(["git", "-C", str(work), "push", "-q", remote, "old"], check=True)
    assert len(load_results(results_repo=remote, revision="old").results) == 1
    with pytest.raises(ValueError, match="is a clone of"):
        load_results(results_repo=str(tmp_path / "other.git"))
