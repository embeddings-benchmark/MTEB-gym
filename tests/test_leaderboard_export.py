"""Tests for leaderboard.export, and for the app on what it exports, on hand-written records in the
results repository's layout: no mteb, no network."""

import importlib.util
import json
import shutil
import sys
import types
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
    """A record in the results repository's layout: <task>/<task>__<rest>.json."""
    task = rec["task_name"]
    p = root / task / f"{task}__{name}.json"
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
    # records written before the labels baseline carry neither key
    assert "labels_baseline" not in sci and "labels" not in sci and all("ndcg_at_10" not in m for m in sci["models"])


def test_labels_baseline_is_carried_when_present(tmp_path):
    rec = record("SciFact", "synthetic")
    for r, n in zip(rec["ratings"], (0.31234, 0.44321)):
        r["ndcg_at_10"] = n
    rec["labels"] = "seed_documents"
    rec["agreement"] = {
        "spearman_rho": 0.5,
        "labels_baseline": {"labels": "seed_documents", "n_models": 2, "spearman_rho": 1.0, "kendall_tau": 0.99999},
    }
    write(tmp_path, "sci-syn", rec)
    write(tmp_path, "sci-orig", record("SciFact", "original", reliability=GOOD))
    sci = export.build_export(tmp_path)["corpora"]["SciFact"]
    assert sci["labels_baseline"] == {
        "labels": "seed_documents",
        "spearman_rho": 1.0,
        "kendall_tau": 1.0,
        "n_models": 2,
    }
    assert [m["ndcg_at_10"] for m in sci["models"]] == [0.4432, 0.3123]  # in rating order
    assert sci["labels"] == "seed_documents"  # what those numbers are scored against
    # a baseline that could not be computed is left out, not exported as a number
    rec["agreement"]["labels_baseline"] = {"error": "need >=3 shared models, have 2", "shared": []}
    write(tmp_path, "sci-syn", rec)
    assert "labels_baseline" not in export.build_export(tmp_path)["corpora"]["SciFact"]


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
    only = export.build_export(tmp_path, judge="judge-x", generator="other-gen")  # nothing to rank: rows only
    assert only["corpora"] == {} and sorted(only["reliability"]) == ["SciFact"] and only["meta"]["judge"] == "judge-x"
    with pytest.raises(export.ExportError, match="no record matches"):
        export.build_export(tmp_path, judge="nobody")
    with pytest.raises(export.ExportError, match="no records"):
        export.build_export(tmp_path / "empty")


def test_original_arm_records_alone_export_reliability_rows(tmp_path):
    """A clone of the results repository that holds original-arm records only, each scored: no ranking,
    and the row says what the labels were."""
    for task in ("NFCorpus", "ArguAna"):
        rec = record(task, "original", reliability=GOOD)
        rec["labels"] = "dataset"
        write(tmp_path, f"judge-x__original-queries__q100-s0-{task[:4]}", rec)
    out = export.build_export(tmp_path)
    assert out["corpora"] == {} and sorted(out["reliability"]) == ["ArguAna", "NFCorpus"]
    assert out["reliability"]["NFCorpus"]["labels"] == "dataset" and out["meta"]["mteb_version"] == "2.20.0"
    assert out["meta"]["dropped"] == []


def test_an_export_with_nothing_in_it_is_refused(tmp_path):
    for task in ("NFCorpus", "ArguAna"):  # original-arm records that were never scored
        write(tmp_path, f"judge-x__original-queries__q100-s0-{task[:4]}", record(task, "original"))
    with pytest.raises(export.ExportError, match="nothing to export"):
        export.build_export(tmp_path)
    write(tmp_path, "sci-syn", record("SciFact", "synthetic"))  # a ranking with no row, dropped: still nothing
    with pytest.raises(export.ExportError, match="nothing to export"):
        export.build_export(tmp_path, allow_missing=True)


def test_cli_writes_file(tmp_path, capsys):
    write(tmp_path, "sci-syn", record("SciFact", "synthetic"))
    write(tmp_path, "sci-orig", record("SciFact", "original", reliability=GOOD))
    out = tmp_path / "data" / "leaderboard_export.json"
    export.main(["--output-folder", str(tmp_path), "--out", str(out)])
    assert json.loads(out.read_text())["corpora"]["SciFact"]["models"][0]["model"] == "m/a"
    assert "1 ranked corpora, 1 reliability rows" in capsys.readouterr().out


APP = Path(__file__).resolve().parents[1] / "leaderboard" / "app.py"


def load_app(tmp_path: Path, monkeypatch, data: dict) -> list[tuple[str, tuple, dict]]:
    """Import a copy of app.py next to `data`, with gradio stubbed: the widgets it built, in order."""
    pytest.importorskip("pandas")
    made = []

    class Widget:
        def __init__(self, *args, **kwargs):
            made.append((type(self).__name__, args, kwargs))

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def change(self, *args, **kwargs):
            pass

    gr = types.ModuleType("gradio")
    for name in ("Blocks", "Tab", "Markdown", "Dropdown", "Dataframe"):
        setattr(gr, name, type(name, (Widget,), {}))
    monkeypatch.setitem(sys.modules, "gradio", gr)
    monkeypatch.setitem(sys.modules, "spaces", None)  # a local run: no ZeroGPU package
    app_dir = tmp_path / "app"
    (app_dir / "data").mkdir(parents=True)
    (app_dir / "data" / "leaderboard_export.json").write_text(json.dumps(data))
    shutil.copy(APP, app_dir / "app.py")
    spec = importlib.util.spec_from_file_location("leaderboard_app_under_test", app_dir / "app.py")
    spec.loader.exec_module(importlib.util.module_from_spec(spec))
    return made


def test_app_loads_an_export_of_reliability_rows_only(tmp_path, monkeypatch):
    for task in ("NFCorpus", "ArguAna"):
        write(tmp_path, f"judge-x__original-queries__q100-s0-{task[:4]}", record(task, "original", reliability=GOOD))
    made = load_app(tmp_path, monkeypatch, export.build_export(tmp_path))
    assert [args[0] for name, args, _ in made if name == "Tab"] == ["Reliability"]
    assert not any(name == "Dropdown" for name, _, _ in made)
    (frame,) = [kw["value"] for name, _, kw in made if name == "Dataframe"]
    assert sorted(frame["corpus"]) == ["ArguAna", "NFCorpus"]


def test_app_ranks_the_exported_corpora(tmp_path, monkeypatch):
    write(tmp_path, "sci-syn", record("SciFact", "synthetic"))
    write(tmp_path, "sci-orig", record("SciFact", "original", reliability=GOOD))
    made = load_app(tmp_path, monkeypatch, export.build_export(tmp_path))
    assert [args[0] for name, args, _ in made if name == "Tab"] == ["Rankings", "Reliability"]
    (dropdown,) = [kw for name, _, kw in made if name == "Dropdown"]
    assert dropdown["value"] == "SciFact"
    ranking = next(kw["value"] for name, _, kw in made if name == "Dataframe")
    assert list(ranking["model"]) == ["m/a", "m/b"]
