import json
import tempfile
import types
from pathlib import Path

from mteb_gym.cache import grades_key, judge_pair_cached, model_id, verdict_key
from mteb_gym.judge import PairwiseJudge, PointwiseJudge
from mteb_gym.llm import MockLLM
from mteb_gym.queries import Query
from mteb_gym.retrieval import Ranked
from mteb_gym.run import pair_subset
from tests.helpers import Deciding, fake_ranked


def _vkey(judge, top_k, query_set, a, rev_a, b, rev_b):
    return verdict_key(model_id(judge.client), judge.system, top_k, judge.doc_chars, query_set, a, rev_a, b, rev_b)


def _gkey(judge, query_set):
    return grades_key(model_id(judge.client), judge.system, judge.instruction, judge.doc_chars, query_set)


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
        judge = PairwiseJudge(Counting(seed=1), "t", workers=1)
        key = _vkey(judge, 5, "qs", "m_a", "r1", "m_b", "r1")
        full = judge_pair_cached(vdir, judge, "m_a", "m_b", ra, rb, key, "qs")
        assert len(full) == 6 and calls["n"] == 12
        # one file per pair, one line per comparison, each naming its run
        (jsonl,) = vdir.glob("*")
        assert jsonl.suffix == ".jsonl" and len(jsonl.read_text().splitlines()) == 6
        line = json.loads(jsonl.read_text().splitlines()[0])
        assert (line["task"], line["query_set"]) == (vdir.name, "qs") and line["judge"]
        # simulate a crash mid-pair: two verdicts written, a third cut short, then a rerun
        lines = jsonl.read_text().splitlines()
        jsonl.write_text("\n".join(lines[:2]) + "\n" + lines[2][:20])
        calls["n"] = 0
        resumed = judge_pair_cached(vdir, judge, "m_a", "m_b", ra, rb, key)
        assert [v.qid for v in resumed] == [q.qid for q in queries] and calls["n"] == 8, (
            "resume judges only the 4 missing"
        )
        calls["n"] = 0
        assert len(judge_pair_cached(vdir, judge, "m_a", "m_b", ra, rb, key)) == 6 and calls["n"] == 0, (
            "the cut-short line is skipped and nothing written after it is lost"
        )
        # the judge's own settings are part of its identity: thinking mode changes the verdicts
        from mteb_gym.cache import model_id as _model_id

        assert _model_id(MockLLM()) != _model_id(types.SimpleNamespace(model="mock", extra_body={"think": True}))
        # a new revision of one model is a new key: no reuse
        assert _vkey(judge, 5, "qs", "m_a", "r1", "m_b", "r2") != key
        assert _vkey(PairwiseJudge(Counting(seed=1), "t", doc_chars=300), 5, "qs", "m_a", "r1", "m_b", "r1") != key
        calls["n"] = 0
        judge_pair_cached(
            vdir,
            judge,
            "m_a",
            "m_b",
            ra,
            fake_ranked("c", queries),
            _vkey(judge, 5, "qs", "m_a", "r1", "m_b", "r2"),
        )
        assert calls["n"] == 12 and len(list(vdir.glob("*.jsonl"))) == 2
        # a run over a subset of queries, then the full run: each query is judged once in total
        key3 = _vkey(judge, 5, "qs3", "m_a", "r1", "m_b", "r1")
        calls["n"] = 0
        part = judge_pair_cached(vdir, judge, "m_a", "m_b", ra[:2], rb[:2], key3)
        assert [v.qid for v in part] == ["q0", "q1"] and calls["n"] == 4
        whole = judge_pair_cached(vdir, judge, "m_a", "m_b", ra, rb, key3)
        assert [v.qid for v in whole] == [q.qid for q in queries] and calls["n"] == 12
    chosen = pair_subset(10, ["q0", "q1", "q2"], 3, seed=0)
    assert all(len(s) == 3 for s in chosen.values()) and chosen == pair_subset(10, ["q0", "q1", "q2"], 3, seed=0)
    assert pair_subset(10, ["q0"], None, 0) is None and pair_subset(3, ["q0"], 5, 0) is None


def test_pointwise_judge_and_grade_cache():
    from mteb_gym.cache import judge_pool_cached
    from mteb_gym.judge import is_decision_model

    decider = Deciding()
    assert is_decision_model(decider) and not is_decision_model(MockLLM())
    queries = [Query(f"q{i}", f"statins heart {i}", []) for i in range(3)]
    docs = {
        f"D{s}{j}": f"statins and heart disease, part {j}" if s == "a" else f"gardening tip {j}"
        for s in "ab"
        for j in range(4)
    }
    ranked = {"a": fake_ranked("a", queries, k=4), "b": fake_ranked("b", queries, k=4)}
    with tempfile.TemporaryDirectory() as tmp:
        judge = PointwiseJudge(decider, "t", workers=2)
        key = _gkey(judge, "qs")
        grades = judge_pool_cached(Path(tmp), judge, ranked, docs, key, "qs")
        assert len(grades) == 24 and decider.calls == 24  # pool: 2 models x 4 docs x 3 queries, each graded once
        assert all(g.parsed_ok and g.probabilities for g in grades)
        assert min(g.grade for g in grades if g.doc_id.startswith("Da")) > max(
            g.grade for g in grades if g.doc_id.startswith("Db")
        )
        (jsonl,) = Path(tmp).glob("*.jsonl")
        assert json.loads(jsonl.read_text().splitlines()[0])["query_set"] == "qs"
        # a rerun grades nothing; an added model grades only the documents it brings
        decider.calls = 0
        judge_pool_cached(Path(tmp), judge, ranked, docs, key, "qs")
        assert decider.calls == 0
        ranked["c"] = [Ranked(r.qid, r.query, ["Da0", "Dnew"], ["", ""]) for r in ranked["a"]]
        docs["Dnew"] = "statins"
        assert len(judge_pool_cached(Path(tmp), judge, ranked, docs, key, "qs")) == 27 and decider.calls == 3
        # a chat LLM grades with a JSON answer
        chat = PointwiseJudge(MockLLM(), "t", workers=1)
        assert not chat.decision and chat.system != judge.system
        assert _gkey(chat, "qs") != key


def test_source_documents_are_judged_without_becoming_model_results(tmp_path):
    from mteb_gym.cache import judge_pool_cached

    decider = Deciding()
    judge = PointwiseJudge(decider, "t", workers=1)
    ranked = {"m": [Ranked("q", "statins", ["retrieved"], ["gardening"])]}
    docs = {"retrieved": "gardening", "source": "statins"}
    grades = judge_pool_cached(tmp_path, judge, ranked, docs, "key", source_pairs={"q": ["source", "source"]})
    assert {g.doc_id for g in grades} == set(docs)
    assert decider.calls == 2
    assert ranked["m"][0].doc_ids == ["retrieved"]
    judge_pool_cached(tmp_path, judge, ranked, docs, "key", source_pairs={"q": ["source"]})
    assert decider.calls == 2
