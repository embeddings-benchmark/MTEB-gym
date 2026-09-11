"""Tests for analysis.seed_baseline: a hand-built output folder with a known nDCG, then the mock fixtures
end to end when ANALYSIS_FIXTURES points at a real mock_results folder. No mteb, no network."""

import json
import math
import os
from pathlib import Path

import pytest

from analysis import seed_baseline as sb
from mteb_gym import reliability as rel

MODELS = ["m/a", "m/b", "m/c"]
FIXTURES = os.environ.get("ANALYSIS_FIXTURES")


def rank(model: str, seed: str) -> dict[str, float]:
    """Model a puts the seed document at rank 1, b at rank 2, c never (nDCG 1.0 / 1/log2(3) / 0.0)."""
    order = {"m/a": [seed, "x1", "x2"], "m/b": ["x1", seed, "x2"], "m/c": ["x1", "x2", "x3"]}[model]
    return {d: float(len(order) - j) for j, d in enumerate(order)}


def seed_of(q: dict) -> str:
    """The document the models retrieve for `q`: its first seed, or a filler nobody is labelled with."""
    seeds = q["seed_doc_ids"]
    return seeds[0] if seeds and seeds != ["nobody"] else "nobody-has-this"


def write_run(root: Path, queries: list[dict], *, truth_ranking: list[str] | None = None) -> Path:
    """A synthetic-arm run: queries file, one prediction file per model, one record. `queries` rows carry
    qid and seed_doc_ids; a query whose seed is "nobody" is retrieved by no model."""
    config = {
        "arm": "synthetic",
        "query_set": "corpus-gen-abc",
        "judge_model": "judge-x",
        "judge_system": "You compare two retrieval systems.",
        "top_k": 10,
        "models": MODELS,
        "model_revisions": {m: "rev1" for m in MODELS},
        "n_queries": len(queries),
        "config_hash": "abcd1234",
    }
    record = {
        "task_name": "ToyRetrieval",
        "source": "mteb",
        "config": config,
        "diagnostics": {},
        "ratings": [
            {"model": m, "rating": r, "ci_low": r - 10, "ci_high": r + 10}
            for m, r in zip(MODELS, (1010.0, 1000.0, 990.0))
        ],
    }
    if truth_ranking is not None:
        record["agreement"] = {"truth_ranking": truth_ranking, "gym_ranking": MODELS}
    qpath = sb.queries_file(root, record)
    qpath.parent.mkdir(parents=True)
    qpath.write_text(json.dumps({"n_generated": len(queries), "queries": queries}))
    for m in MODELS:
        p = rel.prediction_file(root, record, m)
        p.parent.mkdir(parents=True)
        hits = {q["qid"]: rank(m, seed_of(q)) for q in queries}
        p.write_text(json.dumps({"default": {"test": hits}}))
    path = root / "records" / "ToyRetrieval__judge-x__gen__q1-s0-abcd1234.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(record))
    return path


def test_known_ndcg_one_query(tmp_path):
    path = write_run(tmp_path, [{"qid": "q0", "text": "t", "seed_doc_ids": ["d0"], "quality": 5}])
    report = sb.seed_baseline(tmp_path, path)
    by_model = {r["model"]: r for r in report["models"]}
    assert by_model["m/a"]["seed_ndcg"] == 1.0
    assert by_model["m/b"]["seed_ndcg"] == pytest.approx(1 / math.log2(3))
    assert by_model["m/c"]["seed_ndcg"] == 0.0
    assert report["seed_ranking"] == ["m/a", "m/b", "m/c"]
    assert [r["seed_rank"] for r in report["models"]] == [1, 2, 3]
    assert report["coverage"] == {"n_queries": 1, "n_covered": 1, "fraction": 1.0, "uncovered": []}
    # ratings 1010 > 1000 > 990 order the models the same way as the seed labels
    assert report["gym_ranking"] == ["m/a", "m/b", "m/c"]
    assert report["vs_gym"]["spearman_rho"] == pytest.approx(1.0)
    assert report["vs_gym"]["kendall_tau"] == pytest.approx(1.0)
    # no agreement on the record and no --truth: reported as not computed, never as a number
    assert "error" in report["vs_truth"] and report["truth_ranking"] is None
    assert "error" in report["gym_vs_truth"]


def test_truth_from_record_and_from_file(tmp_path):
    path = write_run(
        tmp_path, [{"qid": "q0", "text": "t", "seed_doc_ids": ["d0"]}], truth_ranking=["m/c", "m/b", "m/a"]
    )
    report = sb.seed_baseline(tmp_path, path)
    assert report["truth_ranking"] == ["m/c", "m/b", "m/a"]
    assert report["truth_source"].startswith("record.agreement.truth_ranking")
    assert report["vs_truth"]["spearman_rho"] == pytest.approx(-1.0)
    assert report["vs_truth"]["kendall_tau"] == pytest.approx(-1.0)
    assert report["gym_vs_truth"]["spearman_rho"] == pytest.approx(-1.0)
    truth = tmp_path / "truth.json"
    truth.write_text(json.dumps({"m/a": 50.0, "m/b": 60.0, "m/c": 40.0}))
    report = sb.seed_baseline(tmp_path, path, truth=truth)
    assert report["truth_ranking"] == ["m/b", "m/a", "m/c"]
    assert report["vs_truth"]["spearman_rho"] == pytest.approx(0.5)
    assert {r["model"]: r["truth_rank"] for r in report["models"]} == {"m/a": 2, "m/b": 1, "m/c": 3}


def test_uncovered_query_counts_as_zero(tmp_path):
    queries = [
        {"qid": "q0", "text": "t", "seed_doc_ids": ["d0"]},
        {"qid": "q1", "text": "t", "seed_doc_ids": ["nobody"]},
        {"qid": "q2", "text": "t", "seed_doc_ids": []},  # no seed document: not scorable, not counted
    ]
    report = sb.seed_baseline(tmp_path, write_run(tmp_path, queries))
    assert report["coverage"] == {"n_queries": 2, "n_covered": 1, "fraction": 0.5, "uncovered": ["q1"]}
    by_model = {r["model"]: r for r in report["models"]}
    assert by_model["m/a"]["seed_ndcg"] == pytest.approx(0.5)
    assert by_model["m/a"]["n_queries_scored"] == 2


def test_refuses_other_arm_and_templates(tmp_path):
    path = write_run(tmp_path, [{"qid": "q0", "text": "t", "seed_doc_ids": ["d0"]}])
    record = json.loads(path.read_text())
    record["config"]["arm"] = "original"
    path.write_text(json.dumps(record))
    with pytest.raises(sb.SeedBaselineError, match="synthetic"):
        sb.seed_baseline(tmp_path, path)
    record["config"]["arm"] = "synthetic"
    path.write_text(json.dumps(record))
    qpath = sb.queries_file(tmp_path, record)
    qpath.write_text(qpath.read_text().replace('"text": "t"', '"text": "${QUERY}"'))
    with pytest.raises(sb.SeedBaselineError, match="template"):
        sb.seed_baseline(tmp_path, path)


def test_constant_seed_scores_render_as_undefined(tmp_path):
    # every model misses the seed: seed nDCG is 0 for all, so a rank correlation is undefined, not a number
    path = write_run(tmp_path, [{"qid": "q0", "text": "t", "seed_doc_ids": ["nobody"]}])
    with pytest.warns(Warning, match="constant"):
        report = sb.seed_baseline(tmp_path, path)
    assert report["coverage"] == {"n_queries": 1, "n_covered": 0, "fraction": 0.0, "uncovered": ["q0"]}
    assert report["vs_gym"]["n_models"] == 3 and report["vs_gym"]["spearman_rho"] is None
    assert "seed vs gym: undefined" in sb.format_markdown(report)


def test_missing_prediction_rows_are_an_error(tmp_path):
    queries = [{"qid": "q0", "text": "t", "seed_doc_ids": ["d0"]}, {"qid": "q1", "text": "t", "seed_doc_ids": ["d1"]}]
    path = write_run(tmp_path, queries)
    record = json.loads(path.read_text())
    p = rel.prediction_file(tmp_path, record, "m/b")
    hits = json.loads(p.read_text())
    del hits["default"]["test"]["q1"]
    p.write_text(json.dumps(hits))
    with pytest.raises(sb.SeedBaselineError, match="q1"):
        sb.seed_baseline(tmp_path, path)
    p.unlink()
    with pytest.raises(sb.SeedBaselineError, match="no predictions for m/b"):
        sb.seed_baseline(tmp_path, path)


def test_correlate_needs_three_shared():
    assert "error" in sb.correlate({"a": 1.0, "b": 2.0}, {"a": 1.0, "b": 2.0})
    assert "error" in sb.correlate({"a": 1.0}, None)
    r = sb.correlate({"a": 1.0, "b": 2.0, "c": 3.0}, {"a": 3.0, "b": 2.0, "c": 1.0, "d": 9.0})
    assert r["n_models"] == 3 and r["spearman_rho"] == pytest.approx(-1.0)


def test_main_writes_json_and_markdown(tmp_path, capsys):
    path = write_run(
        tmp_path, [{"qid": "q0", "text": "t", "seed_doc_ids": ["d0"]}], truth_ranking=["m/a", "m/b", "m/c"]
    )
    out = tmp_path / "report" / "seed_baseline.json"
    sb.main(["--output-folder", str(tmp_path), "--record", str(path), "--out", str(out)])
    report = json.loads(out.read_text())
    assert report["seed_ranking"] == ["m/a", "m/b", "m/c"] and report["vs_truth"]["spearman_rho"] == 1.0
    md = out.with_suffix(".md").read_text()
    assert "| 1 | m/a | 1.0000 | 1010.0 (1) | 3.00 (1) |" in md
    assert "coverage: 1/1 queries" in md
    assert "seed vs official: Spearman 1.000" in capsys.readouterr().out


# ----------------------------------------------------------------------------- the mock fixtures
@pytest.mark.skipif(not FIXTURES, reason="ANALYSIS_FIXTURES not set")
def test_fixture_end_to_end(tmp_path):
    root = Path(FIXTURES)
    synthetic = [
        p
        for p in sorted((root / "records").glob("*.json"))
        if json.loads(p.read_text())["config"]["arm"] == "synthetic"
    ]
    assert len(synthetic) == 1, synthetic
    report = sb.seed_baseline(root, synthetic[0])
    record = json.loads(synthetic[0].read_text())
    assert report["task_name"] == record["task_name"]
    assert report["n_models"] == len(record["ratings"]) == 3
    assert set(report["seed_ranking"]) == {r["model"] for r in record["ratings"]}
    assert report["gym_ranking"] == [r["model"] for r in sorted(record["ratings"], key=lambda r: -r["rating"])]
    n_queries = len(json.loads(sb.queries_file(root, record).read_text())["queries"])
    cov = report["coverage"]
    assert cov["n_queries"] == n_queries and 0 <= cov["n_covered"] <= n_queries
    assert all(0.0 <= r["seed_ndcg"] <= 1.0 for r in report["models"])
    assert -1.0 <= report["vs_gym"]["spearman_rho"] <= 1.0
    # the mock record's agreement failed (no official scores offline), so truth is absent until --truth
    assert "error" in report["vs_truth"]
    truth = tmp_path / "truth.json"
    truth.write_text(json.dumps({r["model"]: float(i) for i, r in enumerate(record["ratings"])}))
    with_truth = sb.seed_baseline(root, synthetic[0], truth=truth)
    assert with_truth["vs_truth"]["n_models"] == 3 and with_truth["truth_source"].startswith("--truth")
    out = tmp_path / "seed_baseline.json"
    sb.main(["--output-folder", str(root), "--record", str(synthetic[0]), "--out", str(out)])
    assert out.exists() and out.with_suffix(".md").exists()
