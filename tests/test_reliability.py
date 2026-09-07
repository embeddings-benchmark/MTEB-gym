"""Tests for mteb_gym.reliability on a synthetic output folder: no mteb, no network."""

import json
import math
from pathlib import Path

import pytest

from mteb_gym import reliability as rel
from mteb_gym.results import Result

MODELS = ["m/a", "m/b", "m/c"]
QRELS = {f"q{i}": {f"d{i}": 1} for i in range(8)}  # one relevant doc per query; q8 has no label


def rank(model: str, qid: str) -> dict[str, float]:
    """Model a always finds the label at rank 1, b at rank 2, c never (nDCG 1.0 / 0.63 / 0.0)."""
    i = int(qid[1:])
    order = {"m/a": [f"d{i}", "x1", "x2"], "m/b": ["x1", f"d{i}", "x2"], "m/c": ["x1", "x2", "x3"]}[model]
    return {d: float(len(order) - j) for j, d in enumerate(order)}


def write_run(root: Path, verdicts: dict[tuple[str, str], dict[str, float]], arm: str = "original") -> Path:
    config = {
        "arm": arm,
        "query_set": "corpus-original",
        "judge_model": "judge-x",
        "judge_system": "You compare two retrieval systems.",
        "top_k": 10,
        "models": MODELS,
        "model_revisions": {m: "rev1" for m in MODELS},
        "n_queries": 9,
    }
    record = {
        "task_name": "ToyRetrieval",
        "source": "mteb",
        "config": config,
        "diagnostics": {},
        "ratings": [{"model": m, "rating": 1000.0, "ci_low": 990.0, "ci_high": 1010.0} for m in MODELS],
    }
    for m in MODELS:
        p = rel.prediction_file(root, record, m)
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps({"default": {"test": {f"q{i}": rank(m, f"q{i}") for i in range(9)}}}))
    for (a, b), scores in verdicts.items():
        p = rel.verdict_file(root, record, a, b)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps([{"qid": q, "query": q, "model_a": a, "model_b": b, "score_a": s} for q, s in scores.items()])
        )
    path = root / "records" / "ToyRetrieval__judge-x__corpus-original__q9-s0-abcd1234.json"
    path.parent.mkdir(parents=True)
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


def test_verdict_and_prediction_paths_match_run():
    from mteb_gym.run import verdict_key

    class Client:
        model = "judge-x"

    class Judge:
        client = Client()
        system = "You compare two retrieval systems."

    record = {
        "task_name": "T",
        "config": {
            "judge_model": "judge-x",
            "judge_system": Judge.system,
            "top_k": 10,
            "query_set": "qs",
            "model_revisions": {"m/a": "r1", "m/b": None},
        },
    }
    key = verdict_key(Judge(), 10, "qs", "m/a", "r1", "m/b", None)
    assert rel.verdict_file(Path("out"), record, "m/a", "m/b") == Path("out") / "verdicts" / f"m_a__m_b-{key}.json"
    assert rel.prediction_file(Path("out"), record, "m/b") == Path("out/predictions/m_b@None/qs/T_predictions.json")


def test_judge_reliability_end_to_end(tmp_path):
    # judge agrees with the labels on every committed comparison, ties on q0, and splits (0.5) on q1 of one pair
    perfect = {q: 1.0 for q in QRELS}  # a beats b, a beats c, b beats c, all with A shown first
    verdicts = {
        ("m/a", "m/b"): {**perfect, "q0": 0.5},
        ("m/a", "m/c"): {**perfect, "q1": 0.5},
        ("m/b", "m/c"): {**perfect, "q8": 1.0},  # q8 has no label: must be counted as missing, not scored
    }
    path = write_run(tmp_path, verdicts)
    out = rel.judge_reliability(path, tmp_path, qrels=QRELS, bootstrap=200, seed=0)
    assert "error" not in out
    assert out["n_comparisons"] == 25 and out["n_missing_ndcg"] == 1
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
    out = rel.judge_reliability(path, tmp_path, qrels=QRELS, bootstrap=300, seed=1, write=False)
    assert out["committed_agreement"] == pytest.approx(16 / 24)
    assert out["s_committed"] == pytest.approx(2 * 16 / 24 - 1)
    assert out["cells"] == {"AA": 16, "AB": 8, "BA": 0, "BB": 0}
    assert out["cohen_kappa_committed"] == 0.0  # judge always says A: no information beyond the marginal
    lo, hi = out["s_committed_ci95"]
    assert lo <= out["s_committed"] <= hi
    assert "reliability" not in json.loads(path.read_text())  # write=False leaves the record alone
    # the same seed reproduces the interval exactly; a different seed need not
    again = rel.judge_reliability(path, tmp_path, qrels=QRELS, bootstrap=300, seed=1, write=False)
    assert again["s_committed_ci95"] == out["s_committed_ci95"]


def test_reverse_pair_order_and_jsonl_resume(tmp_path):
    right = {q: 1.0 for q in QRELS}
    path = write_run(tmp_path, {("m/a", "m/b"): right, ("m/a", "m/c"): right})
    record = json.loads(path.read_text())
    # the third pair was written with the models the other way round, and only its .jsonl stream exists
    p = rel.verdict_file(tmp_path, record, "m/c", "m/b").with_suffix(".jsonl")
    p.write_text(
        "\n".join(json.dumps({"qid": q, "query": q, "model_a": "m/c", "model_b": "m/b", "score_a": 0.0}) for q in QRELS)
        + "\n"
    )
    out = rel.judge_reliability(path, tmp_path, qrels=QRELS, bootstrap=50, write=False)
    assert out["n_comparisons"] == 24 and out["committed_agreement"] == 1.0


def test_errors_are_explicit(tmp_path):
    path = write_run(tmp_path, {("m/a", "m/b"): {q: 1.0 for q in QRELS}}, arm="synthetic")
    assert "original-query arm" in rel.judge_reliability(path, tmp_path)["error"]
    out = rel.judge_reliability(path, tmp_path, qrels=QRELS)  # explicit qrels: any arm, but a pair file is missing
    assert "no verdicts" in out["error"]
    assert "reliability" not in json.loads(path.read_text())
    ties = {("m/a", "m/b"): {q: 0.5 for q in QRELS}, ("m/a", "m/c"): {}, ("m/b", "m/c"): {}}
    path2 = write_run(tmp_path / "two", ties)
    assert "no decisive" in rel.judge_reliability(path2, tmp_path / "two", qrels=QRELS)["error"]


def test_reliability_all_scores_only_original_arm(tmp_path):
    right = {q: 1.0 for q in QRELS}
    full = {("m/a", "m/b"): right, ("m/a", "m/c"): right, ("m/b", "m/c"): right}
    write_run(tmp_path / "orig", full)
    write_run(tmp_path / "synth", full, arm="synthetic")
    outs = rel.reliability_all(tmp_path, qrels=QRELS, bootstrap=20, write=False)
    assert len(outs) == 1 and next(iter(outs)).endswith("abcd1234.json")
    assert next(iter(outs.values()))["s_committed"] == 1.0
