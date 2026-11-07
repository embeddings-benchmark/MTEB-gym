import json
import tempfile
import types
from functools import partial
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
    mteb = __import__("mteb")
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
            task_instruction="Given a topic, retrieve documents about it",
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
        assert len(files["judgments"]) == 1 and all(p.exists() for p in files["judgments"] + files["queries"])
        assert rec["config"]["n_queries"] == 4 and rec["config"]["task_instruction"]  # local corpus: no task prompt
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
        assert own.record["config"]["query_source"] == "own" and own.record["config"]["n_queries"] == 3
        assert own.record["labels"] is None and all(r["ndcg_at_10"] is None for r in own.record["ratings"])
        df = load_results(Path(tmp) / "out").to_dataframe()
        assert len(df) == 4 and set(df["query_source"]) == {"synthetic", "own"}


def test_end_to_end_pointwise_decision_judge():
    __import__("mteb")
    from mteb_gym import run

    with tempfile.TemporaryDirectory() as tmp:
        docs = Path(tmp) / "docs"
        docs.mkdir()
        for did, text in make_corpus(12).items():
            (docs / f"{did}.txt").write_text(text)
        decider = Deciding()
        kw = dict(
            task_instruction="Given a topic, retrieve documents about it",
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
        assert len(files["judgments"]) == 1 and files["judgments"][0].exists()  # the grades file
        decider.calls = 0
        assert run(docs, **kw).record == rec and decider.calls == 0  # cached
        pairwise = run(docs, judging="pairwise", **kw).record
        assert pairwise["config"]["judging"] == "pairwise" and pairwise["ratings"] != rec["ratings"]
        assert pairwise["diagnostics"]["parse_failure_rate"] == 0
        assert pairwise["diagnostics"]["judge_calls"] == decider.calls > 0
        decider.calls = 0
        assert run(docs, judging="pairwise", **kw).record == pairwise and decider.calls == 0  # cached


def test_predict_then_run_reuses_the_predictions():
    """One model per call, as a roster is run one model per process; the later run finds the
    prediction files already written and only judges."""
    __import__("mteb")
    from mteb_gym import predict, run

    models = ["mteb/baseline-bm25s", "sentence-transformers/all-MiniLM-L6-v2"]
    with tempfile.TemporaryDirectory() as tmp:
        docs = Path(tmp) / "docs"
        docs.mkdir()
        for did, text in make_corpus(12).items():
            (docs / f"{did}.txt").write_text(text)
        shared = dict(
            task_instruction="Given a topic, retrieve documents about it",
            doc_chars=2000,
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

        res = run(
            docs,
            models,
            judge=Counting(),
            generator=MockLLM(),
            output_folder=Path(tmp) / "out",
            **shared,
        )
        assert len(res.record["ratings"]) == 2 and calls["n"] > 0  # judged
        found = set((Path(tmp) / "cache" / "predictions").rglob("*_predictions.json"))
        assert found == set(paths)  # the run wrote no new prediction files


def test_end_to_end_mteb_task():
    __import__("mteb")
    from mteb_gym import run

    run = partial(run, judging="pairwise")

    with tempfile.TemporaryDirectory() as tmp:
        res = run(
            "NanoNFCorpusRetrieval",
            ["mteb/baseline-bm25s", "sentence-transformers/all-MiniLM-L6-v2"],
            judge=MockLLM(),
            task_instruction="Given a health topic, retrieve documents about it",
            n_queries=8,
            output_folder=Path(tmp),
            cache_folder=Path(tmp) / "cache",
            workers=1,
        )
        cfg = res.record["config"]
        assert len(res.record["ratings"]) == 2 and cfg["n_queries"] >= 5  # the mock's queries survive filtering
        assert "retrieve" in cfg["task_instruction"]
        assert res.record["diagnostics"]["n_comparisons"] == cfg["n_queries"]
        assert res.record["source"] == "mteb" and "@" in res.record["corpus_id"] and cfg["query_set"]


def _docs(tmp, n=30):
    docs = Path(tmp) / "docs"
    docs.mkdir(exist_ok=True)
    for i in range(n):
        (docs / f"{i}.txt").write_text(f"document {i} about topic {i % 5} and words alpha beta gamma {i * 7}")
    return docs


class Calls(MockLLM):
    def __init__(self, seed=0):
        super().__init__(seed)
        self.prompts = []

    def chat(self, messages, **kw):
        self.prompts.append(messages)
        return super().chat(messages, **kw)


_BM25 = ["mteb/baseline-bm25s"]
_TWO = ["mteb/baseline-bm25s", "sentence-transformers/all-MiniLM-L6-v2"]


def test_the_generator_writes_and_the_judge_judges(tmp_path):
    from mteb_gym import run

    run = partial(run, judging="pairwise")

    gen, judge = Calls(), Calls()
    run(
        _docs(tmp_path),
        _TWO,
        judge=judge,
        generator=gen,
        task_instruction="t",
        n_queries=4,
        filter_queries=False,
        output_folder=tmp_path / "o",
        cache_folder=tmp_path / "c",
        workers=1,
    )
    assert all("write realistic" in p[0]["content"] for p in gen.prompts)
    assert judge.prompts and all("compare two retrieval systems" in p[0]["content"] for p in judge.prompts)


def test_pairs_per_query_limits_the_comparisons(tmp_path):
    from mteb_gym import run

    run = partial(run, judging="pairwise")

    three = [*_TWO, "BAAI/bge-small-en-v1.5"]
    judge = Calls()
    res = run(
        _docs(tmp_path),
        three,
        judge=judge,
        generator=MockLLM(),
        task_instruction="t",
        n_queries=4,
        filter_queries=False,
        pairs_per_query=1,
        output_folder=tmp_path / "o",
        cache_folder=tmp_path / "c",
        workers=1,
    )
    assert res.record["diagnostics"]["n_comparisons"] == 4  # one pair for each of the four queries


def test_model_order_does_not_change_what_is_judged(tmp_path):
    from mteb_gym import run

    kw = dict(
        judge=Calls(),
        generator=MockLLM(),
        task_instruction="t",
        n_queries=3,
        filter_queries=False,
        output_folder=tmp_path / "o",
        cache_folder=tmp_path / "c",
        workers=1,
    )
    first = run(_docs(tmp_path), _TWO, **kw)
    calls = len(kw["judge"].prompts)
    second = run(_docs(tmp_path), list(reversed(_TWO)), **kw)
    assert len(kw["judge"].prompts) == calls and first.path == second.path


def test_original_queries_retrieve_and_score_only_the_sample(tmp_path, monkeypatch):
    """The labels baseline and the judge must see the same queries, and only those are retrieved."""
    from mteb_gym import corpus as corpus_mod
    from mteb_gym import run

    docs = {f"d{i}": f"document {i} about alpha beta" for i in range(20)}
    queries = {f"q{i}": f"alpha question {i}" for i in range(10)}
    qrels = {f"q{i}": {f"d{i}": 1} for i in range(10)}
    local = corpus_mod.load(_docs(tmp_path, 3))
    local.docs, local.queries, local.qrels = docs, queries, qrels
    monkeypatch.setattr(corpus_mod, "load", lambda spec: local)
    res = run(
        "x",
        _BM25 + _TWO[1:],
        judge=MockLLM(),
        queries="original",
        n_queries=4,
        task_instruction="t",
        output_folder=tmp_path / "o",
        cache_folder=tmp_path / "c",
        workers=1,
    )
    assert res.record["config"]["n_queries"] == 4 and res.record["labels"] == "dataset"
    predictions = cache_files(res.record, tmp_path / "c")["predictions"][0]
    assert len(json.loads(predictions.read_text())["default"]["test"]) == 4


def test_a_query_with_no_relevant_document_is_not_sampled(tmp_path, monkeypatch):
    from mteb_gym.run import resolve_queries

    corp = types.SimpleNamespace(
        top_ranked=None,
        name="t",
        id="t@1",
        docs={},
        queries={"q0": "a", "q1": "b"},
        qrels={"q0": {"d": 1}, "q1": {"d": 0}},
    )
    gen = types.SimpleNamespace()
    assert list(resolve_queries(corp, tmp_path, "original", gen, 5, 0).texts) == ["q0"]


def test_a_rerun_after_a_scoring_change_rewrites_the_record(tmp_path, monkeypatch):
    import sys

    from mteb_gym import run

    run = partial(run, judging="pairwise")

    run_module = sys.modules["mteb_gym.run"]  # `mteb_gym.run`, the attribute, is the function

    kw = dict(
        judge=MockLLM(),
        generator=MockLLM(),
        task_instruction="t",
        n_queries=3,
        filter_queries=False,
        output_folder=tmp_path / "o",
        cache_folder=tmp_path / "c",
        workers=1,
    )
    first = run(_docs(tmp_path), _TWO, **kw)
    first.record["agreement"] = {"kept": True}
    first.to_disk()
    assert run(_docs(tmp_path), _TWO, **kw).record["agreement"] == {"kept": True}  # nothing changed: the record stands
    real = run_module.rate

    def shifted(*a, **k):
        out = real(*a, **k)
        for r in out:
            r.rating += 1
        return out

    monkeypatch.setattr(run_module, "rate", shifted)
    again = run(_docs(tmp_path), _TWO, **kw).record
    assert "agreement" not in again and again["ratings"][0]["rating"] == first.record["ratings"][0]["rating"] + 1


def test_the_instruction_reaches_the_pairwise_judge(tmp_path):
    from mteb_gym import run

    judge = Calls()
    run(
        _docs(tmp_path),
        _TWO,
        judge=judge,
        generator=MockLLM(),
        task_instruction="my special task",
        n_queries=3,
        filter_queries=False,
        output_folder=tmp_path / "o",
        cache_folder=tmp_path / "c",
        workers=1,
    )
    assert judge.prompts and all("my special task" in p[0]["content"] for p in judge.prompts)


def test_a_run_whose_judge_failed_writes_no_record(tmp_path):
    from mteb_gym import run

    class Broken(MockLLM):
        def chat(self, messages, **kw):
            return "not json"

    with pytest.raises(RuntimeError):
        run(
            _docs(tmp_path),
            _TWO,
            judge=Broken(),
            generator=MockLLM(),
            task_instruction="t",
            n_queries=3,
            filter_queries=False,
            output_folder=tmp_path / "o",
            cache_folder=tmp_path / "c",
            workers=1,
        )
    assert not list((tmp_path / "o").rglob("*.json"))


def _prediction_of(record, cache):
    """Each rated model's prediction file, in the order of the ratings."""
    by_model = {p.parts[-3].split("@")[0]: p for p in cache_files(record, cache)["predictions"]}
    return [by_model[r["model"].replace("/", "_")] for r in record["ratings"]]


def test_each_models_no_judge_baseline_is_recorded(tmp_path):
    from mteb_gym import retrieval, run

    res = run(
        _docs(tmp_path),
        _TWO,
        judge=MockLLM(),
        generator=MockLLM(),
        task_instruction="t",
        n_queries=3,
        filter_queries=False,
        output_folder=tmp_path / "o",
        cache_folder=tmp_path / "c",
        workers=1,
    )
    assert res.record["labels"] == "seed_documents"
    assert all(r["ndcg_at_10"] > 0 for r in res.record["ratings"])
    queries = json.loads(cache_files(res.record, tmp_path / "c")["queries"][0].read_text())["queries"]
    qrels = {q["qid"]: {d: 1 for d in q["seed_doc_ids"]} for q in queries}
    for rating, path in zip(res.record["ratings"], _prediction_of(res.record, tmp_path / "c"), strict=True):
        assert rating["ndcg_at_10"] == pytest.approx(retrieval.ndcg_at_10(path, qrels))


def _excluding_corpus(tmp_path):
    from mteb_gym import corpus as corpus_mod

    docs = tmp_path / "docs"
    docs.mkdir()
    texts = {
        "copy": "statins heart disease statins heart disease",
        "good": "statins and the heart",
        "other": "gardening in the spring",
        "more": "fibre and the gut",
    }
    for did, text in texts.items():
        (docs / f"{did}.txt").write_text(text)
    corp = corpus_mod.load(docs)
    corp.queries = {"q": "statins heart disease"}
    corp.qrels = {"q": {"good.txt": 1}}
    corp.top_ranked = {"q": ["good.txt", "other.txt", "more.txt"]}
    return corp


def test_a_tasks_excluded_documents_stay_out_of_its_own_queries_results(tmp_path):
    """Scored on `good.txt`: first of what `q` may retrieve, so nDCG@10 is 1. With `copy.txt` in the results it is second, and
    nDCG@10 is 1 / log2(3) = 0.631."""
    from mteb_gym import retrieval, task

    corp = _excluding_corpus(tmp_path)
    pred = retrieval.predict(
        "mteb/baseline-bm25s", task.build(corp, None, corp.queries), tmp_path / "with", batch_size=2
    )
    assert retrieval.top_k(pred, corp, corp.queries, 1)[0].doc_ids == ["good.txt"]
    assert "copy.txt" not in retrieval.hits(pred)["q"]
    assert retrieval.ndcg_at_10(pred, corp.qrels) == pytest.approx(1.0)
    corp.top_ranked = None
    pred = retrieval.predict(
        "mteb/baseline-bm25s", task.build(corp, None, corp.queries), tmp_path / "without", batch_size=2
    )
    assert retrieval.top_k(pred, corp, corp.queries, 1)[0].doc_ids == ["copy.txt"]
    assert retrieval.ndcg_at_10(pred, corp.qrels) == pytest.approx(0.6309, abs=1e-3)


def test_only_a_tasks_own_queries_are_limited_and_their_limit_is_part_of_their_identity(tmp_path):
    from mteb_gym.queries import Query
    from mteb_gym.run import resolve_queries
    from mteb_gym.task import build

    corp = _excluding_corpus(tmp_path)
    only_asked = corp.top_ranked
    corp.top_ranked = {**only_asked, "not-asked": ["good.txt"]}  # the task's list covers queries this run does not ask
    own = build(corp, None, corp.queries)
    own.load_data()
    assert own.dataset["default"]["test"]["top_ranked"] == only_asked
    new = build(corp, [Query("q", "statins", ["good.txt"])], {"q": "statins"})  # the same id, not the task's query
    new.load_data()
    assert new.dataset["default"]["test"]["top_ranked"] is None
    limited = resolve_queries(corp, tmp_path, "original", None, 1, 0).id
    corp.top_ranked = None
    assert resolve_queries(corp, tmp_path, "original", None, 1, 0).id != limited


def test_loading_a_task_keeps_its_documents_per_query(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import mteb

    from mteb_gym import corpus as corpus_mod

    data = {
        "corpus": [{"id": "a", "title": "", "text": "x"}],
        "queries": [{"id": "q", "text": "y"}],
        "relevant_docs": {"q": {"a": 1}},
        "top_ranked": {"q": ["a"]},
    }
    task = SimpleNamespace(
        load_data=lambda: None,
        convert_v1_dataset_format_to_v2=lambda num_proc: None,
        dataset={"default": {"standard": data}},
        metadata=MagicMock(dataset={"path": "p", "revision": "r"}, eval_langs=["eng"]),
        ignore_identical_ids=False,
    )
    monkeypatch.setattr(mteb, "get_tasks", lambda tasks: [task])
    assert corpus_mod.load("not-a-path-xyz").top_ranked == {"q": ["a"]}


def test_a_pairwise_run_shows_the_generator_and_the_judge_the_pairwise_limit(tmp_path):
    from mteb_gym import MockLLM, run

    docs = tmp_path / "docs"
    docs.mkdir()
    for did, text in make_corpus(8).items():
        (docs / f"{did}.txt").write_text(text)
    rec = run(
        docs,
        ["mteb/baseline-bm25s", "sentence-transformers/all-MiniLM-L6-v2"],
        MockLLM(),
        generator=MockLLM(),
        n_queries=2,
        judging="pairwise",
        output_folder=tmp_path / "out",
        cache_folder=tmp_path / "cache",
        workers=1,
    ).record
    assert rec["config"]["doc_chars"] == 4000 and rec["config"]["gen_doc_chars"] == 16000
    assert rec["config"]["task_instruction"] is None and rec["config"]["gen_filter"] is False


def test_the_sentence_reaches_the_generator_and_the_judge(tmp_path):
    from mteb_gym import run

    class Recording(MockLLM):
        def __init__(self):
            super().__init__()
            self.systems = []

        def chat(self, messages, **kw):
            self.systems.append(messages[0]["content"])
            return super().chat(messages, **kw)

    docs = tmp_path / "docs"
    docs.mkdir()
    for did, text in make_corpus(12).items():
        (docs / f"{did}.txt").write_text(text)

    def go(sentence):
        generator, judge = Recording(), Recording()
        rec = run(
            docs,
            ["mteb/baseline-bm25s"],
            judge,
            generator,
            task_instruction=sentence,
            n_queries=3,
            judging="pointwise",
            output_folder=tmp_path / "out",
            cache_folder=tmp_path / f"cache-{sentence}",
            workers=1,
        ).record
        return generator, judge, rec

    generator, judge, rec = go("find refutations")
    assert generator.systems and all("find refutations" in s for s in generator.systems)
    assert judge.systems and all("find refutations" in s for s in judge.systems)
    assert rec["config"]["task_instruction"] == "find refutations"
    plain_generator, plain_judge, plain = go(None)
    assert not any("retrieval task" in s for s in plain_generator.systems + plain_judge.systems)
    assert rec["config"]["query_set"] != plain["config"]["query_set"]  # it changes what is generated


def test_a_cap_is_part_of_what_a_run_identifies_and_the_embedding_cache_is_not(tmp_path):
    __import__("mteb")
    from mteb_gym import run

    docs = tmp_path / "docs"
    docs.mkdir()
    for did, text in make_corpus(12).items():
        (docs / f"{did}.txt").write_text(text)
    shared = dict(
        models=["mteb/baseline-bm25s", "sentence-transformers/all-MiniLM-L6-v2"],
        judge=MockLLM(),
        queries=["statins and heart disease", "fiber and the gut"],
        judging="pairwise",
        output_folder=tmp_path / "out",
        workers=1,
    )
    plain = run(docs, cache_folder=tmp_path / "plain", **shared).record
    cached = run(docs, cache_folder=tmp_path / "cached", embedding_cache=tmp_path / "vectors", **shared).record
    assert "max_seq_length" not in plain["config"]
    assert cached["config"] == plain["config"]  # the same run: only where its vectors came from differs
    assert [r["rating"] for r in cached["ratings"]] == pytest.approx([r["rating"] for r in plain["ratings"]])
    assert (tmp_path / "vectors" / "sentence-transformers_all-MiniLM-L6-v2").is_dir()
    assert not (tmp_path / "vectors" / "mteb_baseline-bm25s").exists()

    capped = run(docs, cache_folder=tmp_path / "plain", max_seq_length=64, **shared).record
    assert (
        capped["config"]["max_seq_length"] == 64 and capped["config"]["config_hash"] != plain["config"]["config_hash"]
    )
    files = cache_files(capped, tmp_path / "plain")
    assert all(p.exists() for p in files["predictions"] + files["judgments"])  # named from the record alone
    assert all("-max64" in str(p.parent.parent) for p in files["predictions"])
    assert not set(files["predictions"]) & set(cache_files(plain, tmp_path / "plain")["predictions"])
    assert not set(files["judgments"]) & set(cache_files(plain, tmp_path / "plain")["judgments"])


def test_run_and_predict_pass_the_cache_and_the_cap_on_to_retrieval(tmp_path, monkeypatch):
    __import__("mteb")
    from mteb_gym import predict, retrieval, run

    docs = tmp_path / "docs"
    docs.mkdir()
    for did, text in make_corpus(6).items():
        (docs / f"{did}.txt").write_text(text)
    real, seen = retrieval.predict, []

    def spy(model, task, folder, **kw):
        seen.append((kw["embedding_cache"], kw["max_seq_length"]))
        return real(model, task, folder, batch_size=kw["batch_size"])

    monkeypatch.setattr(retrieval, "predict", spy)
    shared = dict(
        queries=["statins and heart disease"],
        cache_folder=tmp_path / "c",
        embedding_cache="vectors",
        max_seq_length=64,
        workers=1,
    )
    path = predict(docs, "mteb/baseline-bm25s", **shared)
    rec = run(docs, ["mteb/baseline-bm25s"], MockLLM(), judging="pointwise", output_folder=tmp_path / "out", **shared)
    assert seen == [("vectors", 64), ("vectors", 64)]
    assert "-max64" in path.parent.parent.name and cache_files(rec.record, tmp_path / "c")["predictions"] == [path]


def test_a_decision_model_is_refused_as_a_generator(tmp_path):
    from mteb_gym import predict, run

    class Decides:
        model = "d"

        def decide(self, kind, state, question, options=None):
            return {}

    (tmp_path / "a.txt").write_text("text")
    with pytest.raises(ValueError, match="cannot write queries"):
        run(tmp_path, ["m"], MockLLM(), Decides(), output_folder=tmp_path / "o", cache_folder=tmp_path / "c")
    with pytest.raises(ValueError, match="cannot write queries"):
        predict(tmp_path, "m", generator=Decides(), cache_folder=tmp_path / "c")


def test_a_run_prints_its_judge_and_generator_without_any_logging_set_up(tmp_path, capsys):
    from mteb_gym import run

    docs = tmp_path / "docs"
    docs.mkdir()
    for did, text in make_corpus(12).items():
        (docs / f"{did}.txt").write_text(text)
    run(
        docs,
        ["mteb/baseline-bm25s", "sentence-transformers/all-MiniLM-L6-v2"],
        MockLLM(),
        "mock",
        n_queries=4,
        output_folder=tmp_path / "o",
        cache_folder=tmp_path / "c",
        workers=1,
    )
    out = capsys.readouterr().out
    assert "judge: mock (your own object)\n" in out and "generator: mock (mock model)\n" in out


@pytest.mark.parametrize("judging", ["pointwise", "pairwise"])
def test_a_record_is_built_again_from_cached_judgments_with_no_server_and_no_download(tmp_path, monkeypatch, judging):
    """The final build of a large run: the same record, from the cache, by a judge named the same that is never asked."""
    import httpx

    from mteb_gym import run
    from mteb_gym.models.jev import JevServer as DecisionModel

    docs = tmp_path / "docs"
    docs.mkdir()
    for did, text in make_corpus(12).items():
        (docs / f"{did}.txt").write_text(text)
    kw = dict(
        models=["mteb/baseline-bm25s", "sentence-transformers/all-MiniLM-L6-v2"],
        generator=MockLLM(),
        task_instruction="Given a topic, retrieve documents about it",
        n_queries=4,
        judging=judging,
        output_folder=tmp_path / "o",
        cache_folder=tmp_path / "c",
        workers=1,
    )
    judged = Deciding()
    judged.model = "org/jev"
    first = run(docs, judge=judged, **kw)
    assert judged.calls > 0

    def nothing_is_fetched(*a, **k):
        pytest.fail("a cached run fetched something")

    monkeypatch.setattr("mteb_gym.local.hf_hub_download", nothing_is_fetched)
    monkeypatch.setattr(httpx.Client, "send", nothing_is_fetched)
    again = run(docs, judge=DecisionModel("org/jev", "http://unused/v1", api_key="k"), **kw)
    assert again.record == first.record and again.path == first.path
