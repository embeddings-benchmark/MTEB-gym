"""Tests for analysis.scaling: a hand-built run with a known answer, and the mock fixtures end to end."""

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pytest

from analysis import scaling as sc
from mteb_gym import reliability as rel

MODELS = ["m/a", "m/b", "m/c", "m/d"]  # a always beats b beats c beats d
N_QUERIES = 12


def write_run(root: Path) -> Path:
    """Transitive, noiseless verdicts: the stronger model wins every query of every pair."""
    config = {
        "arm": "synthetic",
        "query_set": "corpus-mock-abc",
        "judge_model": "judge-x",
        "judge_system": "You compare two retrieval systems.",
        "top_k": 10,
        "models": MODELS,
        "model_revisions": {m: "rev1" for m in MODELS},
        "n_queries": N_QUERIES,
        "config_hash": "abcd1234",
    }
    record = {
        "task_name": "ToyRetrieval",
        "source": "mteb",
        "config": config,
        "ratings": [{"model": m, "rating": 1000.0, "ci_low": 990.0, "ci_high": 1010.0} for m in MODELS],
        "agreement": {"truth_ranking": list(MODELS)},
    }
    for i, a in enumerate(MODELS):
        for b in MODELS[i + 1 :]:
            a_first = (i + MODELS.index(b)) % 2 == 0  # some pairs are stored in the other order
            p = rel.verdict_file(root, record, a, b) if a_first else rel.verdict_file(root, record, b, a)
            p.parent.mkdir(parents=True, exist_ok=True)
            rows = [
                {"qid": f"q{q}", "query": f"q{q}", "model_a": a, "model_b": b, "score_a": 1.0}
                if a_first
                else {"qid": f"q{q}", "query": f"q{q}", "model_a": b, "model_b": a, "score_a": 0.0}
                for q in range(N_QUERIES)
            ]
            p.write_text(json.dumps(rows))
    path = root / "records" / "ToyRetrieval__judge-x__mock__q12-s0-abcd1234.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(record))
    return path


def test_truth_is_ordinal_from_the_record():
    record = {"config": {"models": ["x", "y", "z"]}, "agreement": {"truth_ranking": ["z", "x", "y"]}}
    assert sc.load_truth(record) == {"x": 2.0, "y": 1.0, "z": 3.0}
    assert sc.load_truth(record, {"x": 0.1, "y": 0.9, "z": 0.5}) == {"x": 0.1, "y": 0.9, "z": 0.5}
    with pytest.raises(ValueError, match="truth_ranking"):
        sc.load_truth({"config": {"models": ["x"]}, "agreement": {"error": "offline"}})
    with pytest.raises(ValueError, match="no truth score"):
        sc.load_truth(record, {"x": 1.0})


def test_axis_values_cap_at_the_record_and_end_at_full():
    axes = sc.axis_values(12, 4, sc.QUERY_GRID, sc.PAIR_FRACTIONS, sc.MIN_MODELS)
    assert axes == {"queries": [5, 10, 12], "pairs": [0.25, 0.5, 1.0], "models": [4]}
    axes = sc.axis_values(100, 7, sc.QUERY_GRID, sc.PAIR_FRACTIONS, sc.MIN_MODELS)
    assert axes["queries"] == [5, 10, 20, 40, 80, 100] and axes["models"] == [4, 5, 6, 7]
    assert sc.axis_values(3, 3, sc.QUERY_GRID, sc.PAIR_FRACTIONS, sc.MIN_MODELS)["models"] == [3]


def test_summarize_counts_degenerate_draws():
    s = sc.summarize([1.0, 0.9, None, 1.0], full_rho=1.0)
    assert s["n_valid"] == 3 and s["n_draws"] == 4
    assert s["mean"] == pytest.approx(2.9 / 3) and s["within"] == pytest.approx(2 / 3)
    assert s["p2.5"] == pytest.approx(0.905) and s["p97.5"] == pytest.approx(1.0)
    empty = sc.summarize([None, None], full_rho=1.0)
    assert empty["mean"] is None and empty["n_valid"] == 0 and empty["n_draws"] == 2


def test_transitive_verdicts_agree_at_every_size(tmp_path):
    record = write_run(tmp_path)
    r = sc.scaling(tmp_path, record, draws=30, seed=1)
    assert r["n_models"] == 4 and r["n_pairs"] == 6 and r["n_queries"] == N_QUERIES
    assert r["full"]["rho"] == 1.0 and r["full"]["gym_ranking"] == MODELS and r["full"]["truth_ranking"] == MODELS
    assert r["truth_source"] == "agreement.truth_ranking (ordinal)"
    assert [p["n_queries"] for p in r["grid"]["queries"]] == [5, 10, 12]
    for p in r["grid"]["queries"] + r["grid"]["models"] + [r["grid"]["pairs"][-1]]:
        assert p["n_valid"] == 30 and p["mean"] == 1.0 and p["p2.5"] == 1.0 and p["p97.5"] == 1.0
        assert p["within"] == 1.0
    assert [p["n_pairs"] for p in r["grid"]["pairs"]] == [2, 3, 6]
    # two of six pairs leave a model unmentioned unless they are disjoint (3 of the 15 choices): draws are dropped
    assert r["grid"]["pairs"][0]["n_valid"] < 30
    assert "| queries | 5 | 6 | 4 | 1.000 |" in r["table"]


def test_reversed_truth_gives_minus_one(tmp_path):
    record = write_run(tmp_path)
    reversed_truth = {m: float(i) for i, m in enumerate(MODELS)}  # d is best, a is worst
    r = sc.scaling(tmp_path, record, truth=reversed_truth, draws=20, seed=0)
    assert r["truth_source"] == "truth file" and r["full"]["rho"] == -1.0
    assert r["full"]["truth_ranking"] == MODELS[::-1]
    for p in r["grid"]["queries"] + r["grid"]["models"] + [r["grid"]["pairs"][-1]]:
        assert p["mean"] == -1.0 and p["within"] == 1.0


def test_subsample_draws_are_seeded_and_sized(tmp_path):
    record = json.loads(write_run(tmp_path).read_text())
    pairs = sc.load_pairs(tmp_path, record)
    assert sorted(len(v) for v in pairs.values()) == [N_QUERIES] * 6
    qids = sc.qids_of(pairs)
    a, chosen_a = sc.subsample(pairs, MODELS, qids, 5, 0.5, 3, np.random.default_rng(3))
    b, chosen_b = sc.subsample(pairs, MODELS, qids, 5, 0.5, 3, np.random.default_rng(3))
    assert chosen_a == chosen_b and [v.qid for v in a] == [v.qid for v in b]
    assert len(chosen_a) == 3 and len({v.qid for v in a}) == 5
    assert len({(v.model_a, v.model_b) for v in a}) == 2  # ceil(0.5 * 3) pairs


def test_tied_ratings_are_dropped_without_a_warning():
    ties = [sc.Verdict(f"q{q}", f"q{q}", a, b, 0.5) for q in range(4) for a, b in [("x", "y"), ("y", "z"), ("x", "z")]]
    truth = {"x": 3.0, "y": 2.0, "z": 1.0}
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # scipy's ConstantInputWarning would fail the test
        assert sc.refit_rho(ties, ["x", "y", "z"], truth) is None
    assert sc.refit_rho(ties[:4], ["x", "y", "z"], truth) is None  # z never judged: unrated
    assert sc.refit_rho(ties, ["x", "y"], truth) is None  # fewer than 3 models


def test_missing_verdict_file_is_an_error(tmp_path):
    record = json.loads(write_run(tmp_path).read_text())
    victim = rel.verdict_file(tmp_path, record, "m/a", "m/b")
    if not victim.exists():
        victim = rel.verdict_file(tmp_path, record, "m/b", "m/a")
    victim.unlink()
    with pytest.raises(FileNotFoundError, match="no verdicts"):
        sc.load_pairs(tmp_path, record)


def test_partial_jsonl_stream_is_read(tmp_path):
    record = json.loads(write_run(tmp_path).read_text())
    for a, b in [("m/a", "m/b"), ("m/b", "m/a")]:
        p = rel.verdict_file(tmp_path, record, a, b)
        if p.exists():
            rows = json.loads(p.read_text())[:7]  # the run stopped after 7 queries on this pair
            p.with_suffix(".jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            p.unlink()
    pairs = sc.load_pairs(tmp_path, record)
    assert sorted(len(v) for v in pairs.values()) == [7] + [N_QUERIES] * 5
    assert sc.qids_of(pairs) == sorted(f"q{q}" for q in range(N_QUERIES))


def test_scaling_rejects_bad_inputs(tmp_path):
    record = json.loads(write_run(tmp_path).read_text())
    with pytest.raises(ValueError, match="at least 3 models"):
        sc.scaling(tmp_path, record | {"config": record["config"] | {"models": MODELS[:2]}}, draws=1)
    with pytest.raises(ValueError, match="min_models"):
        sc.scaling(tmp_path, record, draws=1, min_models=2)
    with pytest.raises(ValueError, match="query counts"):
        sc.scaling(tmp_path, record, draws=1, query_grid=[0, 5])
    with pytest.raises(ValueError, match="pair fractions"):
        sc.scaling(tmp_path, record, draws=1, pair_fractions=[0.5, 1.5])
    with pytest.raises(ValueError, match="pair fractions"):
        sc.scaling(tmp_path, record, draws=1, pair_fractions=[0.0])


def test_main_writes_json_and_prints_the_table(tmp_path, capsys):
    record = write_run(tmp_path)
    out = tmp_path / "scaling.json"
    sc.main(["--output-folder", str(tmp_path), "--record", str(record), "--out", str(out), "--draws", "5"])
    data = json.loads(out.read_text())
    assert data["full"]["rho"] == 1.0 and set(data["grid"]) == {"queries", "pairs", "models"}
    printed = capsys.readouterr().out
    assert "| axis | queries | pairs | models |" in printed and str(out) in printed


@pytest.mark.skipif(not os.environ.get("ANALYSIS_FIXTURES"), reason="ANALYSIS_FIXTURES not set")
def test_mock_fixtures_end_to_end(tmp_path):
    root = Path(os.environ["ANALYSIS_FIXTURES"])
    records = [
        p for p in (root / "records").glob("*.json") if json.loads(p.read_text())["config"]["arm"] == "synthetic"
    ]
    assert len(records) == 1, records
    record = json.loads(records[0].read_text())
    models = record["config"]["models"]
    truth = {m: float(i) for i, m in enumerate(models)}  # the fixture's agreement block is an offline error
    r = sc.scaling(root, records[0], truth=truth, draws=10, seed=0)
    assert r["n_models"] == len(models) and r["n_pairs"] == len(models) * (len(models) - 1) // 2
    assert r["n_queries"] == record["config"]["n_queries"]
    assert -1.0 <= r["full"]["rho"] <= 1.0
    assert r["grid"]["models"] == [r["grid"]["models"][0]] and r["grid"]["models"][0]["n_models"] == len(models)
    for p in r["grid"]["queries"] + r["grid"]["pairs"] + r["grid"]["models"]:
        assert p["n_draws"] == 10 and (p["mean"] is None or -1.0 <= p["mean"] <= 1.0)
    json.dumps(r)  # serialisable as written by main()
