"""Tests for mteb_gym on the mock LLM: no network, no GPU, no API key. The
end-to-end test needs mteb, bm25s and sentence-transformers and is skipped otherwise."""

import json
import tempfile
import types
from pathlib import Path

import pytest

from mteb_gym import Result, agreement, load_results, results
from mteb_gym.judge import Judge, Verdict, judge_system, task_prompt
from mteb_gym.llm import MockLLM
from mteb_gym.queries import Query, QueryGenerator, extract_json
from mteb_gym.rank import format_leaderboard, rate
from mteb_gym.retrieval import Ranked
from mteb_gym.run import judge_pair_cached, pair_subset, resolve_description, verdict_key


def make_corpus(n=40):
    topics = ["heart disease and statins", "vitamin D and asthma", "gut microbiome and fiber", "telomeres and stress"]
    return {f"D{i}": f"Document {i} about {topics[i % 4]}. " + "clinical evidence " * (i % 5 + 1) for i in range(n)}


def fake_ranked(seed, queries, k=5):
    return [
        Ranked(q.qid, q.text, [f"D{seed}{j}" for j in range(k)], [f"result {seed}-{q.qid}-{j}" for j in range(k)])
        for q in queries
    ]


def test_extract_json():
    assert extract_json('```json\n{"winner": "A"}\n```')["winner"] == "A"
    assert extract_json('blah {"score": 4} trailing')["score"] == 4
    assert extract_json("not json") == {}
    assert extract_json('Let me think {"winner": "B"} ... final {"winner": "A"}')["winner"] == "A"  # last object wins
    assert extract_json('{"reasoning": "set {x} wins", "winner": "B"}')["winner"] == "B"


def test_query_generation():
    corpus = make_corpus()
    gen = QueryGenerator(MockLLM(), n_queries=8, filter=True, workers=1)
    kept = gen.run(corpus)
    assert gen.n_generated >= 8 and len(kept) <= 8 and all(q.quality is not None for q in kept)

    # worker count must never change the query set, including under flaky calls
    class Flaky(MockLLM):
        def chat(self, messages, **kw):
            prompt = " ".join(m.get("content", "") for m in messages)
            return "no json here" if self._hash(prompt) % 3 == 0 else super().chat(messages, **kw)

    def gen_with(workers, client):
        return QueryGenerator(client, n_queries=12, filter=False, workers=workers).generate(corpus)

    for client in (MockLLM, Flaky):
        seq, par = gen_with(1, client()), gen_with(4, client())
        assert len(seq) == 12
        assert [(q.qid, q.text, tuple(q.seed_doc_ids)) for q in par] == [
            (q.qid, q.text, tuple(q.seed_doc_ids)) for q in seq
        ]


def test_llm_drops_rejected_params():
    """A model that refuses a parameter (400 naming it) still answers."""
    from mteb_gym.llm import LLM

    class Refused(Exception):
        status_code = 400

    calls = []

    class Completions:
        def create(self, **kw):
            calls.append(sorted(k for k in kw if k in ("temperature", "max_completion_tokens")))
            for k in ("temperature", "max_completion_tokens"):
                if k in kw:
                    raise Refused(f"Unsupported parameter: '{k}' is not supported with this model.")
            return types.SimpleNamespace(
                model="m-2026-01-01", choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="ok"))]
            )

    llm = LLM("m", api_key="x", max_tokens=512, temperature=0.0)
    llm.client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=Completions()))
    assert llm.chat([{"role": "user", "content": "hi"}]) == "ok"
    assert llm.chat([{"role": "user", "content": "hi"}]) == "ok"
    assert calls == [["max_completion_tokens", "temperature"], ["max_completion_tokens"], [], []]  # learned once
    assert llm.served_model == "m-2026-01-01" and llm.sent == {}  # both refused: the record shows neither


def test_description_is_what_the_encoders_get():
    """The judge is given the task's own mteb prompt, the same sentence mteb gives an
    instruction-tuned encoder, or nothing when the task has none."""
    mteb = pytest.importorskip("mteb")
    from mteb_gym.corpus import Corpus

    def described(name):
        return resolve_description(None, Corpus(name, name, {}, mteb.get_task(name).metadata))

    assert described("ArguAna") == ("Given a claim, find documents that refute the claim", "mteb:task_prompt")
    assert described("ClimateFEVERHardNegatives") == (None, None)  # no prompt: so is the encoder's
    assert described("BrightBiologyRetrieval")[0].startswith("Represent this biology post")


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


def test_doc_chars_reaches_the_judge():
    from mteb_gym.judge import Judge

    assert Judge(MockLLM(), doc_chars=300).doc_chars == 300
    assert "x" * 301 not in Judge(MockLLM(), doc_chars=300).system  # the setting is the judge's, not the prompt's


def test_judge():
    from mteb_gym.judge import _format

    shown = _format(Ranked("q", "query", ["d1", "d2"], ["one two three four five", "short"]), doc_chars=12)
    assert shown.splitlines() == ["  1. one two...", "  2. short"]  # cut at a word boundary and marked, or whole
    queries = [Query(f"q{i}", f"query {i} about statins", ["D0"]) for i in range(30)]
    ra, rb = fake_ranked("a", queries), fake_ranked("b", queries)
    seq = Judge(MockLLM(seed=1), workers=1).judge_all(ra, rb, "m_a", "m_b")
    par = Judge(MockLLM(seed=1), workers=8).judge_all(ra, rb, "m_a", "m_b")
    assert all(0.0 <= v.score_a <= 1.0 for v in seq)
    assert [(v.qid, v.score_a) for v in par] == [(v.qid, v.score_a) for v in seq]

    class Exploding:
        def chat(self, messages, **kw):
            raise AssertionError("judge must not be called for identical result sets")

    v = Judge(Exploding()).judge_all(ra[:1], fake_ranked("a", queries[:1]), "m_a", "m_b")[0]
    assert v.score_a == 0.5 and v.raw == ["identical"]

    class Garbage:
        def chat(self, messages, **kw):
            return "I refuse to answer in the requested format."

    verdicts = Judge(Garbage()).judge_all(ra[:5], rb[:5], "m_a", "m_b")
    assert all(v.score_a == 0.5 for v in verdicts)
    assert results.verdict_diagnostics(verdicts)["parse_failure_rate"] == 1.0


def test_rank():
    def v(qid, a, b, score):
        return Verdict(qid=qid, query="q", model_a=a, model_b=b, score_a=score)

    verdicts = []
    for i in range(20):
        verdicts += [
            v(f"q{i}", "winner", "mid", 0.75),
            v(f"q{i}", "mid", "loser", 1.0),
            v(f"q{i}", "winner", "loser", 1.0),
        ]
    ratings = rate(verdicts, bootstrap=100)
    assert [r.name for r in ratings] == ["winner", "mid", "loser"]
    assert ratings[-1].rating < ratings[1].rating - 50, "a model that loses every verdict must sink"
    assert all(r.ci >= 0 for r in ratings)
    assert format_leaderboard(ratings).count("\n") == 6


def test_correlate():
    g = {f"m{i}": float(i) for i in range(5)}
    res = agreement.correlate(g, dict(g), bootstrap=50)
    assert abs(res["spearman_rho"] - 1.0) < 1e-9
    import numpy as np

    truth = {f"m{i}": float(i) for i in range(25)}
    top = list(range(15, 25))
    np.random.default_rng(0).shuffle(top)
    gym = {f"m{i}": float(v) for i, v in zip(range(15, 25), top)} | {f"m{i}": float(i) for i in range(15)}
    out = agreement.correlate(gym, truth, bootstrap=0)
    assert out["spearman_rho"] > 0.85 and abs(out["spearman_top10"]) < 0.6  # strong models shuffled among themselves
    a = np.arange(10, dtype=float)
    assert agreement._tau_ap(a, a) == 1.0 and agreement._tau_ap(-a, a) == -1.0


def test_instruction():
    assert task_prompt("Represent this biology post for searching relevant passages: ") == (
        "Represent this biology post for searching relevant passages:"
    )  # mteb's own text, verbatim; replace it with task_description if it reads badly
    assert (
        task_prompt({"query": "Given a claim, find documents that refute the claim"})
        == "Given a claim, find documents that refute the claim"
    )
    assert task_prompt(None) is None
    corpus = types.SimpleNamespace(
        metadata=types.SimpleNamespace(prompt="Given a claim, find documents that refute the claim")
    )
    assert resolve_description(None, corpus) == (
        "Given a claim, find documents that refute the claim",
        "mteb:task_prompt",
    )
    assert resolve_description("Prefer replies that resolve the ticket", corpus)[1] == "config:task_description"
    bare = types.SimpleNamespace(metadata=types.SimpleNamespace(prompt=None, name="X", adapted_from=None))
    assert resolve_description(None, bare) == (None, None)
    gen = QueryGenerator(MockLLM(), task_description="Given a claim, find documents that refute the claim")
    assert "refute the claim" in gen.system and gen.params["task_description"]  # part of the query cache key
    assert "retrieval task is" not in QueryGenerator(MockLLM()).system
    with_task = judge_system("Given a claim, find documents that refute the claim")
    # the sentence is inserted as written, whatever punctuation it ends with
    assert "relevant passages:\n" in judge_system("Represent this biology post for searching relevant passages:")
    without = judge_system(None)
    assert "refute the claim" in with_task and "retrieval task is" not in without
    # the only difference is the sentence itself, so an arm with one compares with an arm without
    assert (
        with_task.replace("The retrieval task is: Given a claim, find documents that refute the claim\n", "") == without
    )


def test_verdict_cache():
    calls = {"n": 0}

    class Counting(MockLLM):
        def chat(self, messages, **kw):
            calls["n"] += 1
            return super().chat(messages, **kw)

    queries = [Query(f"q{i}", f"query {i}", ["D0"]) for i in range(6)]
    ra, rb = fake_ranked("a", queries), fake_ranked("b", queries)
    with tempfile.TemporaryDirectory() as tmp:
        vdir = Path(tmp)
        judge = Judge(Counting(seed=1), workers=1)
        key = verdict_key(judge, 5, "qs", "m_a", "r1", "m_b", "r1")
        full = judge_pair_cached(vdir, judge, "m_a", "m_b", ra, rb, key)
        assert len(full) == 6 and calls["n"] == 12
        # simulate a crash mid-pair: keep two verdicts in the JSONL, drop the final file, rerun
        final = next(vdir.glob("*.json"))
        jsonl = final.with_suffix(".jsonl")
        jsonl.write_text("\n".join(jsonl.read_text().splitlines()[:2]) + "\n")
        final.unlink()
        calls["n"] = 0
        resumed = judge_pair_cached(vdir, judge, "m_a", "m_b", ra, rb, key)
        assert [v.qid for v in resumed] == [q.qid for q in queries] and calls["n"] == 8, (
            "resume judges only the 4 missing"
        )
        # the judge's own settings are part of its identity: thinking mode changes the verdicts
        from mteb_gym.run import _model_id

        assert _model_id(MockLLM()) != _model_id(types.SimpleNamespace(model="mock", extra_body={"think": True}))
        # a new revision of one model is a new key: no reuse
        assert verdict_key(judge, 5, "qs", "m_a", "r1", "m_b", "r2") != key
        assert verdict_key(Judge(Counting(seed=1), doc_chars=300), 5, "qs", "m_a", "r1", "m_b", "r1") != key
        calls["n"] = 0
        judge_pair_cached(
            vdir,
            judge,
            "m_a",
            "m_b",
            ra,
            fake_ranked("c", queries),
            verdict_key(judge, 5, "qs", "m_a", "r1", "m_b", "r2"),
        )
        assert calls["n"] == 12 and len(list(vdir.glob("*.json"))) == 2
        # a run over a subset of queries, then the full run: each query is judged once in total
        key3 = verdict_key(judge, 5, "qs3", "m_a", "r1", "m_b", "r1")
        calls["n"] = 0
        part = judge_pair_cached(vdir, judge, "m_a", "m_b", ra[:2], rb[:2], key3)
        assert [v.qid for v in part] == ["q0", "q1"] and calls["n"] == 4
        whole = judge_pair_cached(vdir, judge, "m_a", "m_b", ra, rb, key3)
        assert [v.qid for v in whole] == [q.qid for q in queries] and calls["n"] == 12
    chosen = pair_subset(10, ["q0", "q1", "q2"], 3, seed=0)
    assert all(len(s) == 3 for s in chosen.values()) and chosen == pair_subset(10, ["q0", "q1", "q2"], 3, seed=0)
    assert pair_subset(10, ["q0"], None, 0) is None and pair_subset(3, ["q0"], 5, 0) is None


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
        == Path("out") / "records" / f"demo__judge__gen__q3-s0-{exp['config_hash']}.json"
    )
    with tempfile.TemporaryDirectory() as tmp:
        r = Result(rec, Path(tmp) / "records" / "demo.json")
        r.to_disk()
        again = Result.from_disk(r.path)
        assert again.record == rec and "demo" not in again.leaderboard and "a" in again.leaderboard
        df = load_results(tmp).to_dataframe()
        assert list(df["model"]) == [x["model"] for x in rec["ratings"]] and set(df["task"]) == {"demo"}


def test_agreement():
    original = agreement.fetch_truth
    agreement.fetch_truth = lambda models, task, **kw: (
        {"model_a": 30.0, "model_b": 20.0, "model_c": 10.0},
        {m: {"model_revision": "r1", "official": True} for m in models},
    )
    try:
        with tempfile.TemporaryDirectory() as tmp:
            rec = {
                "task_name": "NFCorpus",
                "source": "mteb",
                "config": {},
                "diagnostics": {},
                "ratings": [
                    {"model": m, "rating": r} for m, r in (("model_a", 1100), ("model_b", 1000), ("model_c", 900))
                ],
            }
            res = Result(rec, Path(tmp) / "records" / "NFCorpus.json")
            res.to_disk()
            agr = res.agreement(bootstrap=100, seed=0)
            assert agr["spearman_rho"] == 1.0 and agr["kendall_tau"] == 1.0
            assert Result.from_disk(res.path).record["agreement"]["truth_source"] == {
                m: {"model_revision": "r1", "official": True} for m in ("model_a", "model_b", "model_c")
            }
            assert load_results(tmp).agreement(bootstrap=10)[str(res.path)]["spearman_rho"] == 1.0
            assert (
                Result({"source": "local", "task_name": "x", "ratings": []})
                .agreement()["error"]
                .startswith("local corpus")
            )
    finally:
        agreement.fetch_truth = original


def test_official_scores_across_revisions():
    """The results repository files many models' scores under "external", not under the revision
    mteb pins; the lookup must find those, and prefer the pinned one when both exist."""
    mteb = pytest.importorskip("mteb")
    from mteb.results import TaskResult

    meta = mteb.get_model_meta("BAAI/bge-base-en-v1.5")

    def cache_with(revisions):
        tmp = tempfile.mkdtemp()
        for revision, score in revisions:
            folder = Path(tmp) / "results" / meta.model_name_as_path() / revision
            folder.mkdir(parents=True)
            (folder / "model_meta.json").write_text(json.dumps({"name": meta.name, "revision": revision}))
            TaskResult(
                task_name="NFCorpus",
                dataset_revision="x",
                mteb_version="2.20.10",
                evaluation_time=1.0,
                scores={"test": [{"main_score": score, "hf_subset": "default", "languages": ["eng-Latn"]}]},
            ).to_disk(folder / "NFCorpus.json")
        return mteb.ResultCache(cache_path=tmp)

    external_only = cache_with([("external", 0.40)])
    assert external_only.load_task_result("NFCorpus", meta) is None  # mteb's per-revision lookup: nothing
    assert agreement.official_scores(external_only, "NFCorpus", [meta.name]) == {meta.name: (0.40, "external")}

    both = cache_with([("external", 0.40), (meta.revision, 0.55)])
    assert agreement.official_scores(both, "NFCorpus", [meta.name]) == {meta.name: (0.55, meta.revision)}

    assert agreement.official_scores(external_only, "SciFact", [meta.name]) == {}


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
            models=["mteb/baseline-bm25s", "sentence-transformers/all-MiniLM-L6-v2"],
            judge=Counting(),
            n_queries=4,
            filter_queries=False,
            output_folder=Path(tmp) / "out",
            workers=1,
        )
        res = run(docs, **kw)
        rec = res.record
        assert len(rec["ratings"]) == 2 and res.path.exists() and res.path.parent.name == "records"
        assert (
            rec["config"]["n_queries"] == 4 and rec["config"]["task_description"] is None
        )  # local corpus: no task prompt
        assert rec["source"] == "local" and rec["corpus_id"].startswith("local:docs@")
        assert rec["llms"]["judge"]["model"] == "mock" and rec["llms"]["generator"]["model"] == "mock"
        assert rec["labels"] == "seed_documents" and all(0 <= r["ndcg_at_10"] <= 1 for r in rec["ratings"])
        assert all(
            r["revision"] == mteb.get_model_meta(r["model"]).revision for r in rec["ratings"]
        )  # mteb's pins carried over
        preds = list((Path(tmp) / "out" / "predictions").rglob("*_predictions.json"))
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
        shared = dict(n_queries=4, filter_queries=False, output_folder=Path(tmp) / "out", workers=1)
        with pytest.raises(ValueError, match="generator"):
            predict(docs, models[0], **shared)  # a generated query set is identified by the generator
        paths = [predict(docs, m, generator=MockLLM(), **shared) for m in models]
        assert all(p.exists() for p in paths)

        calls = {"n": 0}

        class Counting(MockLLM):
            def chat(self, messages, **kw):
                calls["n"] += 1
                return super().chat(messages, **kw)

        res = run(docs, models, judge=Counting(), generator=MockLLM(), **shared)
        assert len(res.record["ratings"]) == 2 and calls["n"] > 0  # judged
        found = set((Path(tmp) / "out" / "predictions").rglob("*_predictions.json"))
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
            n_queries=8,
            output_folder=Path(tmp),
            workers=1,
        )
        cfg = res.record["config"]
        assert len(res.record["ratings"]) == 2 and cfg["n_queries"] >= 5  # the mock's queries survive filtering
        assert cfg["task_description_source"] == "mteb:task_prompt" and "retrieve" in cfg["task_description"]
        assert res.record["diagnostics"]["n_comparisons"] == cfg["n_queries"]
        assert res.record["source"] == "mteb" and "@" in res.record["corpus_id"] and cfg["query_set"]
