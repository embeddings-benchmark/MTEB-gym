import json
import tempfile
from pathlib import Path

import pytest

from mteb_gym import cache_files, load_results
from mteb_gym.llm import MockLLM
from tests.helpers import Deciding, make_corpus


def test_original_queries_are_sampled_and_self_matches_dropped():
    """The dataset's own queries obey n_queries, and a query that is itself a document is
    dropped from its own results where mteb's flag says so."""
    from mteb_gym.corpus import Corpus
    from mteb_gym.retrieval import top_k
    from mteb_gym.run import sample_queries

    queries = {f"q{i}": f"query {i}" for i in range(10)}
    assert sample_queries(queries, 20, 0) == queries  # fewer than asked for: all of them
    few = sample_queries(queries, 3, 0)
    assert len(few) == 3 and few == sample_queries(queries, 3, 0)  # the same three every time
    assert list(few) == [q for q in queries if q in few]  # in the dataset's order

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "p.json"
        path.write_text(json.dumps({"default": {"test": {"d0": {"d0": 9.0, "d1": 5.0, "d2": 1.0}}}}))
        docs = {"d0": "the query itself", "d1": "an answer", "d2": "another"}
        plain = Corpus("c", "x@1/default/test", docs, None)
        assert top_k(path, plain, {"d0": "the query itself"}, 2)[0].doc_ids == ["d0", "d1"]
        flagged = Corpus("c", "x@1/default/test", docs, None, ignore_identical_ids=True)
        assert top_k(path, flagged, {"d0": "the query itself"}, 2)[0].doc_ids == ["d1", "d2"]


def test_end_to_end_local_corpus():
    mteb = pytest.importorskip("mteb")
    pytest.importorskip("bm25s")
    pytest.importorskip("sentence_transformers")
    from mteb_gym import run

    calls = {"n": 0}

    class Counting(MockLLM):
        def chat(self, messages, **kw):
            calls["n"] += 1
            return super().chat(messages, **kw)

    with tempfile.TemporaryDirectory() as tmp:
        docs = Path(tmp) / "docs"
        docs.mkdir()
        for did, text in make_corpus(12).items():
            (docs / f"{did}.txt").write_text(text)
        kw = dict(
            task_description="Given a topic, retrieve documents about it",
            models=["mteb/baseline-bm25s", "sentence-transformers/all-MiniLM-L6-v2"],
            judge=Counting(),
            n_queries=4,
            filter_queries=False,
            output_folder=Path(tmp) / "out",
            cache_folder=Path(tmp) / "cache",
            workers=1,
        )
        res = run(docs, **kw)
        rec = res.record
        assert len(rec["ratings"]) == 2 and res.path.exists()
        assert res.path.parent == Path(tmp) / "out" / "docs" and res.path.name.startswith("docs__")  # <task>/<record>
        assert set((Path(tmp) / "out").rglob("*")) == {
            res.path.parent,
            res.path,
        }  # the results folder holds records only
        # the query set keeps the settings of the generator that wrote it; the record reads them from there
        (qfile,) = cache_files(rec, Path(tmp) / "cache")["queries"]
        assert json.loads(qfile.read_text())["generator"]["model"] == "mock" == rec["llms"]["generator"]["model"]
        files = cache_files(rec, Path(tmp) / "cache")
        assert len(files["verdicts"]) == 1 and all(p.exists() for p in files["verdicts"] + files["queries"])
        assert rec["config"]["n_queries"] == 4 and rec["config"]["task_description"]  # local corpus: no task prompt
        assert rec["source"] == "local" and rec["corpus_id"].startswith("local:docs@")
        assert rec["llms"]["judge"]["model"] == "mock" and rec["llms"]["generator"]["model"] == "mock"
        assert rec["labels"] == "seed_documents" and all(0 <= r["ndcg_at_10"] <= 1 for r in rec["ratings"])
        assert all(
            r["revision"] == mteb.get_model_meta(r["model"]).revision for r in rec["ratings"]
        )  # mteb's pins carried over
        preds = list((Path(tmp) / "cache" / "predictions").rglob("*_predictions.json"))
        assert len(preds) == 2 and all("@" in p.parts[-3] for p in preds)  # <model>@<revision>/<query set>/
        assert rec["config"]["model_revisions"] == {r["model"]: r["revision"] for r in rec["ratings"]}
        calls["n"] = 0
        again = run(docs, **kw)  # everything cached: no LLM calls, the record stands as written
        assert calls["n"] == 0 and again.record == rec and again.path == res.path
        own = run(docs, queries=["statins and heart disease", "fiber and the gut", "vitamin D for asthma"], **kw)
        assert own.record["config"]["arm"] == "own" and own.record["config"]["n_queries"] == 3
        assert own.record["labels"] is None and all(r["ndcg_at_10"] is None for r in own.record["ratings"])
        df = load_results(Path(tmp) / "out").to_dataframe()
        assert len(df) == 4 and set(df["arm"]) == {"synthetic", "own"}


def test_end_to_end_pointwise_decision_judge():
    pytest.importorskip("mteb")
    pytest.importorskip("bm25s")
    pytest.importorskip("sentence_transformers")
    from mteb_gym import run

    with tempfile.TemporaryDirectory() as tmp:
        docs = Path(tmp) / "docs"
        docs.mkdir()
        for did, text in make_corpus(12).items():
            (docs / f"{did}.txt").write_text(text)
        decider = Deciding()
        kw = dict(
            task_description="Given a topic, retrieve documents about it",
            models=["mteb/baseline-bm25s", "sentence-transformers/all-MiniLM-L6-v2"],
            judge=decider,
            queries=["statins and heart disease", "fiber and the gut", "vitamin D for asthma"],
            output_folder=Path(tmp) / "out",
            cache_folder=Path(tmp) / "cache",
            workers=1,
        )
        with pytest.raises(ValueError, match="cannot write queries"):
            run(docs, **{**kw, "queries": "synthetic"})
        rec = run(docs, **kw).record
        assert rec["config"]["judging"] == "pointwise" and rec["config"]["pairs_per_query"] is None
        assert len(rec["ratings"]) == 2 and all(0 <= r["rating"] <= 100 for r in rec["ratings"])
        assert rec["diagnostics"]["n_grades"] == decider.calls > 0
        files = cache_files(rec, Path(tmp) / "cache")
        assert len(files["verdicts"]) == 1 and files["verdicts"][0].exists()  # the grades file
        decider.calls = 0
        assert run(docs, **kw).record == rec and decider.calls == 0  # cached
        with pytest.raises(ValueError, match="judges pointwise"):
            run(docs, judging="pairwise", **kw)


def test_predict_then_run_reuses_the_predictions():
    """One model per call, as a roster is run one model per process; the later run finds the
    prediction files already written and only judges."""
    pytest.importorskip("mteb")
    pytest.importorskip("bm25s")
    pytest.importorskip("sentence_transformers")
    from mteb_gym import predict, run

    models = ["mteb/baseline-bm25s", "sentence-transformers/all-MiniLM-L6-v2"]
    with tempfile.TemporaryDirectory() as tmp:
        docs = Path(tmp) / "docs"
        docs.mkdir()
        for did, text in make_corpus(12).items():
            (docs / f"{did}.txt").write_text(text)
        shared = dict(
            task_description="Given a topic, retrieve documents about it",
            n_queries=4,
            filter_queries=False,
            cache_folder=Path(tmp) / "cache",
            workers=1,
        )
        with pytest.raises(ValueError, match="generator"):
            predict(docs, models[0], **shared)  # a generated query set is identified by the generator
        paths = [predict(docs, m, generator=MockLLM(), **shared) for m in models]
        assert all(p.exists() for p in paths)

        calls = {"n": 0}

        class Counting(MockLLM):
            def chat(self, messages, **kw):
                calls["n"] += 1
                return super().chat(messages, **kw)

        res = run(docs, models, judge=Counting(), generator=MockLLM(), output_folder=Path(tmp) / "out", **shared)
        assert len(res.record["ratings"]) == 2 and calls["n"] > 0  # judged
        found = set((Path(tmp) / "cache" / "predictions").rglob("*_predictions.json"))
        assert found == set(paths)  # the run wrote no new prediction files


def test_end_to_end_mteb_task():
    pytest.importorskip("mteb")
    pytest.importorskip("bm25s")
    pytest.importorskip("sentence_transformers")
    from mteb_gym import run

    with tempfile.TemporaryDirectory() as tmp:
        res = run(
            "NanoNFCorpusRetrieval",
            ["mteb/baseline-bm25s", "sentence-transformers/all-MiniLM-L6-v2"],
            judge=MockLLM(),
            task_description="Given a health topic, retrieve documents about it",
            n_queries=8,
            output_folder=Path(tmp),
            cache_folder=Path(tmp) / "cache",
            workers=1,
        )
        cfg = res.record["config"]
        assert len(res.record["ratings"]) == 2 and cfg["n_queries"] >= 5  # the mock's queries survive filtering
        assert cfg["task_description_source"] == "config:task_description" and "retrieve" in cfg["task_description"]
        assert res.record["diagnostics"]["n_comparisons"] == cfg["n_queries"]
        assert res.record["source"] == "mteb" and "@" in res.record["corpus_id"] and cfg["query_set"]
