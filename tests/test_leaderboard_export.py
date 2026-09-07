"""Tests for leaderboard.export on hand-written records: no mteb, no network."""

import json
from pathlib import Path

import pytest

from leaderboard import export


def record(
    task: str, arm: str, *, judge="judge-x", generator="gen-y", config_hash="h1", reliability=None, revision="abc1234"
):
    rec = {
        "task_name": task,
        "source": "mteb",
        "gym_revision": revision,
        "mteb_version": "2.20.0",
        "config": {
            "arm": arm,
            "judge_model": judge,
            "generator_model": generator if arm == "synthetic" else None,
            "n_queries": 50,
            "config_hash": config_hash,
        },
        "diagnostics": {},
        "ratings": [
            {"model": "m/b", "rating": 1010.26, "ci_low": 990.0, "ci_high": 1030.0},
            {"model": "m/a", "rating": 1040.44, "ci_low": 1020.0, "ci_high": 1060.0},
        ],
    }
    if reliability is not None:
        rec["reliability"] = reliability
    return rec


GOOD = {
    "committed_agreement": 0.79512,
    "kappa_committed": 0.59024,
    "s_committed_ci95": [0.41, 0.72],
    "clear_winner_agreement": 0.822,
    "tier": "A",
    "n_models": 2,
    "n_queries_scored": 80,
}


def write(root: Path, name: str, rec: dict) -> None:
    p = root / "records" / f"{name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec))


def test_export_shape(tmp_path):
    write(tmp_path, "sci-syn", record("SciFact", "synthetic"))
    write(tmp_path, "sci-orig", record("SciFact", "original", reliability=GOOD))
    write(tmp_path, "nf-syn", record("NFCorpus", "synthetic", config_hash="h2"))
    write(tmp_path, "nf-orig", record("NFCorpus", "original", reliability={**GOOD, "kappa_committed": 0.565}))
    write(
        tmp_path, "arg-orig", record("ArguAna", "original", reliability={**GOOD, "kappa_committed": 0.151})
    )  # kappa only
    out = export.build_export(tmp_path)
    assert set(out) == {"meta", "corpora", "reliability"}
    assert sorted(out["corpora"]) == ["NFCorpus", "SciFact"]
    assert sorted(out["reliability"]) == ["ArguAna", "NFCorpus", "SciFact"]  # a kappa-only corpus is allowed
    sci = out["corpora"]["SciFact"]
    assert sci["n_queries"] == 50 and sci["config_hash"] == "h1"
    assert [m["model"] for m in sci["models"]] == ["m/a", "m/b"]  # sorted by rating
    assert sci["models"][0] == {"model": "m/a", "rating": 1040.4, "ci_low": 1020.0, "ci_high": 1060.0}
    assert out["reliability"]["SciFact"] == {
        "committed": 0.7951,
        "kappa": 0.5902,
        "kappa_ci95": [0.41, 0.72],
        "clear": 0.822,
        "tier": "A",
        "n_models": 2,
        "n_queries": 80,
    }
    assert out["meta"]["judge"] == "judge-x" and out["meta"]["experiment_commit"] == "abc1234"
    assert out["meta"]["dropped"] == [] and out["meta"]["mteb_version"] == "2.20.0"


def test_missing_reliability_is_refused_unless_dropped(tmp_path):
    write(tmp_path, "sci-syn", record("SciFact", "synthetic"))
    write(tmp_path, "sci-orig", record("SciFact", "original", reliability=GOOD))
    write(tmp_path, "nf-syn", record("NFCorpus", "synthetic"))
    write(tmp_path, "nf-orig", record("NFCorpus", "original", reliability={"error": "no verdicts"}))
    write(tmp_path, "fq-syn", record("FiQA2018", "synthetic"))  # never scored at all
    with pytest.raises(export.ExportError, match="FiQA2018, NFCorpus"):
        export.build_export(tmp_path)
    out = export.build_export(tmp_path, allow_missing=True)
    assert sorted(out["corpora"]) == ["SciFact"] and out["meta"]["dropped"] == ["FiQA2018", "NFCorpus"]


def test_ambiguous_records_need_a_pin(tmp_path):
    write(tmp_path, "sci-orig", record("SciFact", "original", reliability=GOOD))
    write(tmp_path, "sci-syn-1", record("SciFact", "synthetic", config_hash="h1"))
    write(tmp_path, "sci-syn-2", record("SciFact", "synthetic", config_hash="h2"))
    with pytest.raises(export.ExportError, match="2 synthetic records"):
        export.build_export(tmp_path)
    out = export.build_export(tmp_path, pins={"SciFact": "h2"})
    assert out["corpora"]["SciFact"]["config_hash"] == "h2"
    with pytest.raises(export.ExportError, match="config hash h9"):
        export.build_export(tmp_path, pins={"SciFact": "h9"})


def test_judge_and_generator_filters(tmp_path):
    write(tmp_path, "sci-orig", record("SciFact", "original", reliability=GOOD))
    write(tmp_path, "sci-syn-x", record("SciFact", "synthetic", judge="judge-x"))
    write(tmp_path, "sci-syn-z", record("SciFact", "synthetic", judge="judge-z", config_hash="h3"))
    write(tmp_path, "sci-orig-z", record("SciFact", "original", judge="judge-z", reliability=GOOD))
    out = export.build_export(tmp_path, judge="judge-x")
    assert out["corpora"]["SciFact"]["config_hash"] == "h1" and out["meta"]["judge"] == "judge-x"
    with pytest.raises(export.ExportError, match="no synthetic-arm record"):
        export.build_export(tmp_path, judge="judge-x", generator="other-gen")
    with pytest.raises(export.ExportError, match="no records"):
        export.build_export(tmp_path / "empty")


def test_cli_writes_file(tmp_path, capsys):
    write(tmp_path, "sci-syn", record("SciFact", "synthetic"))
    write(tmp_path, "sci-orig", record("SciFact", "original", reliability=GOOD))
    out = tmp_path / "data" / "leaderboard_export.json"
    export.main(["--output-folder", str(tmp_path), "--out", str(out)])
    assert json.loads(out.read_text())["corpora"]["SciFact"]["models"][0]["model"] == "m/a"
    assert "1 ranked corpora, 1 reliability rows" in capsys.readouterr().out
