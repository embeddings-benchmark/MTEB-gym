"""Tests for mteb_gym.reliability on a hand-written results-repo record and cache: no mteb, no network."""

import json
import math
from pathlib import Path

import pytest

from mteb_gym import reliability as rel
from mteb_gym.results import Result
from mteb_gym.retrieval import slug
from mteb_gym.run import _verdict_key, cache_files

TASK = "ToyRetrieval"
QUERY_SET = "toy-original-n9-s0"
SYSTEM = "You compare two retrieval systems."
MODELS = ["m/a", "m/b", "m/c"]
QRELS = {f"q{i}": {f"d{i}": 1} for i in range(8)}  # one relevant doc per query; q8 has no label
OK = [True, True]


def rank(model: str, qid: str) -> dict[str, float]:
    """Model a always finds the label at rank 1, b at rank 2, c never (nDCG 1.0 / 0.63 / 0.0)."""
    i = int(qid[1:])
    order = {"m/a": [f"d{i}", "x1", "x2"], "m/b": ["x1", f"d{i}", "x2"], "m/c": ["x1", "x2", "x3"]}[model]
    return {d: float(len(order) - j) for j, d in enumerate(order)}


def row(a: str, b: str, qid: str, score: float, parsed_ok=OK) -> str:
    v = {"qid": qid, "query": qid, "model_a": a, "model_b": b, "score_a": score, "raw": ["A", "B"]}
    return json.dumps({**v, "parsed_ok": parsed_ok, "task": TASK, "judge": "judge-x", "query_set": QUERY_SET})


def verdict_path(cache: Path, a: str, b: str, doc_chars: int = 300) -> Path:
    key = _verdict_key("judge-x", SYSTEM, 10, doc_chars, QUERY_SET, a, "rev1", b, "rev1")
    return cache / "verdicts" / TASK / f"{slug(a)}__{slug(b)}-{key}.jsonl"


def write_run(root: Path, verdicts: dict[tuple[str, str], dict[str, float]], arm: str = "original") -> Path:
    """A results-repo clone at root/results and a cache at root/cache, written in the layout run() uses:
    results/<task>/<record>.json, cache/predictions/<task>/<model>@<rev>/<query set>/<task>_predictions.json
    and cache/verdicts/<task>/<pair>-<key>.jsonl with one row per comparison."""
    config = {
        "arm": arm,
        "query_set": QUERY_SET,
        "judge_model": "judge-x",
        "judge_system": SYSTEM,
        "top_k": 10,
        "doc_chars": 300,
        "models": MODELS,
        "model_revisions": {m: "rev1" for m in MODELS},
        "n_queries": 9,
        "seed": 0,
        "config_hash": "abcd1234" if arm == "original" else "ef567890",
    }
    record = {
        "task_name": TASK,
        "source": "mteb",
        "config": config,
        "labels": "dataset" if arm == "original" else None,
        "diagnostics": {},
        "ratings": [{"model": m, "rating": 1000.0, "ci_low": 990.0, "ci_high": 1010.0} for m in MODELS],
    }
    cache = root / "cache"
    for m in MODELS:
        p = cache / "predictions" / TASK / f"{slug(m)}@rev1" / QUERY_SET / f"{TASK}_predictions.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"default": {"test": {f"q{i}": rank(m, f"q{i}") for i in range(9)}}}))
    for (a, b), scores in verdicts.items():
        p = verdict_path(cache, a, b)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(row(a, b, q, s) + "\n" for q, s in scores.items()))
    second = "original-queries" if arm == "original" else "gen-y"
    path = root / "results" / TASK / f"{TASK}__judge-x__{second}__q9-s0-{config['config_hash']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))
    return path


def test_ndcg_and_winners():
    assert rel.ndcg_at_k(["d1", "x"], {"d1": 1}, 10) == 1.0
    assert rel.ndcg_at_k(["x", "d1"], {"d1": 1}, 10) == pytest.approx(1 / math.log2(3))
    assert rel.ndcg_at_k(["x"], {"d1": 1}, 10) == 0.0
    assert rel.ndcg_at_k(["d1"], {"d1": 0}, 10) is None  # no positive label: not scorable
    assert rel.ndcg_at_k(["d1", "d2"], {"d1": 1, "d2": 1}, 1) == 1.0  # ideal truncated to k
    assert rel.obj_winner(0.5, 0.5) == "tie" and rel.obj_winner(0.6, 0.5) == "A" and rel.obj_winner(0.5, 0.6) == "B"
    assert rel.judge_winner(1.0) == "A" and rel.judge_winner(0.0) == "B" and rel.judge_winner(0.5) == "tie"
    assert rel.judge_winner(0.75) == "A" and rel.judge_winner(0.25) == "B"  # one order a tie: still a commitment


def test_cell_statistics():
    assert rel.s_of([10, 0, 0, 10]) == 1.0
    assert rel.s_of([5, 5, 5, 5]) == 0.0
    assert rel.s_of([0, 0, 0, 0]) is None
    assert rel.cohen_kappa_of([10, 0, 0, 10]) == 1.0
    assert rel.cohen_kappa_of([10, 0, 0, 0]) is None  # degenerate marginal
    assert rel.cohen_kappa_of([20, 5, 10, 15]) == pytest.approx((0.7 - 0.5) / 0.5)
    assert (
        rel.tier_of(0.45) == "A" and rel.tier_of(0.25) == "B" and rel.tier_of(0.1) == "C" and rel.tier_of(None) is None
    )
    t = rel.assign_tier(0.25, 0.55)
    assert t["tier"] == "B" and t["tier_label"] == "B (CI spans A)" and t["tier_basis"] == "ci_lower_bound"
    assert rel.assign_tier(None, None)["tier"] is None


def test_hand_written_layout_is_the_package_layout(tmp_path):
    """The files this test writes by hand are exactly the ones cache_files names from the record."""
    right = {q: 1.0 for q in QRELS}
    path = write_run(tmp_path, {("m/a", "m/b"): right, ("m/a", "m/c"): right, ("m/b", "m/c"): right})
    files = cache_files(json.loads(path.read_text()), tmp_path / "cache")
    assert files["queries"] == []  # the original arm reads the dataset's queries, none are cached
    assert len(files["predictions"]) == 3 and len(files["verdicts"]) == 3
    assert all(p.exists() for p in files["predictions"] + files["verdicts"])
    assert files["verdicts"][0] == verdict_path(tmp_path / "cache", "m/a", "m/b")


def test_judge_reliability_on_a_gym_run(tmp_path, monkeypatch):
    """run() on the mock LLM, then judge_reliability on its cache: the files it reads are the ones
    run() wrote, so a change to the cache layout fails here and not as a silent "no verdicts" readout."""
    pytest.importorskip("mteb")
    pytest.importorskip("bm25s")
    pytest.importorskip("sentence_transformers")
    from mteb_gym import run
    from mteb_gym.llm import MockLLM

    topics = ["heart disease and statins", "vitamin D and asthma", "gut microbiome and fiber", "telomeres and stress"]
    docs = tmp_path / "docs"
    docs.mkdir()
    for i in range(12):
        (docs / f"D{i}.txt").write_text(f"Document {i} about {topics[i % 4]}. " + "clinical evidence " * (i % 5 + 1))
    out, cache = tmp_path / "out", tmp_path / "cache"
    models = ["mteb/baseline-bm25s", "sentence-transformers/all-MiniLM-L6-v2"]
    res = run(
        docs,
        models,
        judge=MockLLM(),
        n_queries=4,
        filter_queries=False,
        output_folder=out,
        cache_folder=cache,
        workers=1,
    )
    rec = res.record
    assert res.path.parent == out / "docs" and rec["config"]["judge_model"] == "mock"
    files = cache_files(rec, cache)
    assert all(p.exists() for p in files["queries"] + files["predictions"] + files["verdicts"])
    # the labels are the seed documents of each generated query, the ones run() scored ndcg_at_10 against
    (qfile,) = files["queries"]
    qrels = {q["qid"]: {d: 1 for d in q["seed_doc_ids"]} for q in json.loads(qfile.read_text())["queries"]}
    readout = rel.judge_reliability(res, cache, qrels=qrels, bootstrap=50, write=False)
    assert "error" not in readout, readout
    assert readout["n_comparisons"] == rec["config"]["n_queries"] and readout["n_models"] == 2
    assert readout["n_unparsed"] == 0  # the mock judge always answers in form
    assert readout["n_decisive"] + readout["n_missing_ndcg"] <= readout["n_comparisons"]
    assert readout["tier"] in {"A", "B", "C"}
    # the same files through the environment's cache folder, and nowhere else
    monkeypatch.setenv("MTEB_GYM_CACHE", str(cache))
    assert rel.judge_reliability(res, qrels=qrels, bootstrap=50, write=False) == readout
    monkeypatch.setenv("MTEB_GYM_CACHE", str(tmp_path / "elsewhere"))
    assert "no predictions at" in rel.judge_reliability(res, qrels=qrels, write=False)["error"]


def test_judge_reliability_end_to_end(tmp_path):
    # judge agrees with the labels on every committed comparison, ties on q0, and splits (0.5) on q1 of one pair
    perfect = {q: 1.0 for q in QRELS}  # a beats b, a beats c, b beats c, all with A shown first
    verdicts = {
        ("m/a", "m/b"): {**perfect, "q0": 0.5},
        ("m/a", "m/c"): {**perfect, "q1": 0.5},
        ("m/b", "m/c"): {**perfect, "q8": 1.0},  # q8 has no label: must be counted as missing, not scored
    }
    path = write_run(tmp_path, verdicts)
    out = rel.judge_reliability(path, tmp_path / "cache", qrels=QRELS, bootstrap=200, seed=0)
    assert "error" not in out
    assert out["n_comparisons"] == 25 and out["n_missing_ndcg"] == 1 and out["n_unparsed"] == 0
    assert out["n_decisive"] == 24 and out["n_abstain"] == 2 and out["n_committed"] == 22
    assert out["committed_agreement"] == 1.0 and out["s_committed"] == 1.0 and out["kappa_committed"] == 1.0
    assert out["cohen_kappa_committed"] is None  # every committed verdict is AA: degenerate marginal
    assert out["n_clear_winner"] == 24 and out["clear_winner_agreement"] == 1.0
    assert out["tier"] == "A" and out["s_committed_ci95"] == [1.0, 1.0]
    assert out["n_models"] == 3 and out["ndcg_k"] == 10 and out["n_queries_scored"] == 8
    # the readout is persisted in the record, under its own key
    stored = Result.from_disk(path).record["reliability"]
    assert stored["s_committed"] == 1.0 and stored["cells"] == {"AA": 22, "AB": 0, "BA": 0, "BB": 0}


def test_judge_reliability_wrong_judge(tmp_path):
    # judge picks the loser on every comparison of one pair and the winner on the other two: p = 16/24
    flipped = {q: 0.0 for q in QRELS}
    right = {q: 1.0 for q in QRELS}
    path = write_run(tmp_path, {("m/a", "m/b"): flipped, ("m/a", "m/c"): right, ("m/b", "m/c"): right})
    out = rel.judge_reliability(path, tmp_path / "cache", qrels=QRELS, bootstrap=300, seed=1, write=False)
    assert out["committed_agreement"] == pytest.approx(16 / 24)
    assert out["s_committed"] == pytest.approx(2 * 16 / 24 - 1)
    assert out["cells"] == {"AA": 16, "AB": 8, "BA": 0, "BB": 0}
    assert out["cohen_kappa_committed"] == 0.0  # judge always says A: no information beyond the marginal
    lo, hi = out["s_committed_ci95"]
    assert lo <= out["s_committed"] <= hi
    assert "reliability" not in json.loads(path.read_text())  # write=False leaves the record alone
    # the same seed reproduces the interval exactly; a different seed need not
    again = rel.judge_reliability(path, tmp_path / "cache", qrels=QRELS, bootstrap=300, seed=1, write=False)
    assert again["s_committed_ci95"] == out["s_committed_ci95"]


def test_unparsed_comparisons_are_left_out(tmp_path):
    """A comparison where either order failed to parse is not a tie and not a commitment: it is left
    out and counted, as rank.rate leaves it out of the ranking. An identical-retrieval row (no judge
    call, parsed_ok empty) stays a tie, as it does there."""
    right = {q: 1.0 for q in QRELS}
    path = write_run(tmp_path, {("m/a", "m/b"): right, ("m/a", "m/c"): right, ("m/b", "m/c"): right})
    p = verdict_path(tmp_path / "cache", "m/a", "m/b")
    lines = [
        row("m/a", "m/b", "q0", 0.75, [True, False]),  # one order parsed as A, the other failed: would read as A
        row("m/a", "m/b", "q1", 0.25, [False, True]),  # would read as B
        row("m/a", "m/b", "q2", 0.5, [False, False]),  # would read as an abstention
        json.dumps(
            {"qid": "q3", "query": "q3", "model_a": "m/a", "model_b": "m/b", "score_a": 0.5, "raw": ["identical"]}
        ),
    ] + [row("m/a", "m/b", q, 1.0) for q in ("q4", "q5", "q6", "q7")]
    p.write_text("\n".join(lines) + "\n")
    out = rel.judge_reliability(path, tmp_path / "cache", qrels=QRELS, bootstrap=50, write=False)
    assert out["n_comparisons"] == 24 and out["n_unparsed"] == 3
    assert out["n_decisive"] == 21 and out["n_abstain"] == 1 and out["n_committed"] == 20
    assert out["cells"] == {"AA": 20, "AB": 0, "BA": 0, "BB": 0}  # the flipped q1 never reached the cells


def test_cut_lines_and_repeated_queries_read_as_the_package_reads_them(tmp_path):
    right = {q: 1.0 for q in QRELS}
    path = write_run(tmp_path, {("m/a", "m/b"): right, ("m/a", "m/c"): right, ("m/b", "m/c"): right})
    p = verdict_path(tmp_path / "cache", "m/b", "m/c")
    text = p.read_text()
    p.write_text(text + row("m/b", "m/c", "q0", 0.0) + "\n" + "\n" + row("m/b", "m/c", "q1", 1.0)[:40])
    rows = rel.load_verdicts(p)
    assert len(rows) == 8 and rows["q0"].score_a == 0.0 and rows["q1"].score_a == 1.0  # last full row per query
    out = rel.judge_reliability(path, tmp_path / "cache", qrels=QRELS, bootstrap=50, write=False)
    assert out["n_comparisons"] == 24 and out["cells"] == {"AA": 23, "AB": 1, "BA": 0, "BB": 0}


def test_per_query_ndcg_drops_a_query_that_is_its_own_document(tmp_path):
    p = tmp_path / "T_predictions.json"
    p.write_text(json.dumps({"default": {"test": {"q1": {"q1": 3.0, "d1": 2.0, "x": 1.0}}}}))
    assert rel.per_query_ndcg(p, {"q1": {"d1": 1}}, 10) == pytest.approx({"q1": 1 / math.log2(3)})
    assert rel.per_query_ndcg(p, {"q1": {"d1": 1}}, 10, ignore_identical_ids=True) == {"q1": 1.0}


def test_record_dict_is_scored_in_place(tmp_path):
    right = {q: 1.0 for q in QRELS}
    path = write_run(tmp_path, {("m/a", "m/b"): right, ("m/a", "m/c"): right, ("m/b", "m/c"): right})
    record = json.loads(path.read_text())
    out = rel.judge_reliability(record, tmp_path / "cache", qrels=QRELS, bootstrap=20)
    assert record["reliability"] is out and out["s_committed"] == 1.0
    assert "reliability" not in json.loads(path.read_text())  # a dict has no path to write back to


def test_errors_are_explicit(tmp_path):
    cache = tmp_path / "cache"
    path = write_run(tmp_path, {("m/a", "m/b"): {q: 1.0 for q in QRELS}}, arm="synthetic")
    assert "original-query arm" in rel.judge_reliability(path, cache)["error"]
    out = rel.judge_reliability(path, cache, qrels=QRELS)  # explicit qrels: any arm, but a pair file is missing
    assert out["error"].startswith("no verdicts at") and "m_a__m_c-" in out["error"]
    assert "reliability" not in json.loads(path.read_text())
    ties = {("m/a", "m/b"): {q: 0.5 for q in QRELS}, ("m/a", "m/c"): {}, ("m/b", "m/c"): {}}
    path2 = write_run(tmp_path / "two", ties)
    assert "no decisive" in rel.judge_reliability(path2, tmp_path / "two" / "cache", qrels=QRELS)["error"]
    older = json.loads(path2.read_text())
    del older["config"]["doc_chars"]  # written before doc_chars joined the verdict key: not readable here
    assert "predates" in rel.judge_reliability(older, tmp_path / "two" / "cache", qrels=QRELS)["error"]


def test_reliability_all_scores_only_original_arm(tmp_path):
    right = {q: 1.0 for q in QRELS}
    full = {("m/a", "m/b"): right, ("m/a", "m/c"): right, ("m/b", "m/c"): right}
    write_run(tmp_path, full)
    synth = write_run(tmp_path, full, arm="synthetic")  # the same task folder, the same cache
    assert len(list((tmp_path / "results" / TASK).glob("*.json"))) == 2
    outs = rel.reliability_all(tmp_path / "results", tmp_path / "cache", qrels=QRELS, bootstrap=20, write=False)
    assert len(outs) == 1 and next(iter(outs)).endswith("original-queries__q9-s0-abcd1234.json")
    assert next(iter(outs.values()))["s_committed"] == 1.0
    assert "reliability" not in json.loads(synth.read_text())
