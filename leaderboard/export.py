"""
Build data/leaderboard_export.json for the leaderboard app from gym records.

    python -m leaderboard.export --output-folder results --out leaderboard/data/leaderboard_export.json

A corpus's ranking comes from its synthetic-arm record; its reliability row comes from the
original-arm record on the same corpus after mteb_gym.reliability has scored it. The app refuses a
ranked corpus without a reliability row, so this script refuses it first: a ranking without its
kappa is not publishable, and "not measured" must never be shown as a ranking that looks fine.

Records are filtered by judge (and optionally generator); when a corpus still has more than one
synthetic record the script stops and lists their config hashes, and --pin TASK=HASH picks one.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from mteb_gym.results import Result, load_results

RELIABILITY_KEYS = ("committed_agreement", "kappa_committed", "clear_winner_agreement")


class ExportError(Exception):
    pass


def _round(x):
    return None if x is None else round(float(x), 4)


def _ranking(rec: dict) -> dict:
    models = sorted(rec["ratings"], key=lambda r: -r["rating"])
    return {
        "n_queries": rec["config"]["n_queries"],
        "config_hash": rec["config"].get("config_hash"),
        "models": [
            {
                "model": r["model"],
                "rating": round(r["rating"], 1),
                "ci_low": round(r["ci_low"], 1),
                "ci_high": round(r["ci_high"], 1),
            }
            for r in models
        ],
    }


def _reliability_row(rec: dict) -> dict | None:
    r = rec.get("reliability")
    if not r or "error" in r or any(r.get(k) is None for k in RELIABILITY_KEYS):
        return None
    return {
        "committed": _round(r["committed_agreement"]),
        "kappa": _round(r["kappa_committed"]),
        "kappa_ci95": [_round(v) for v in r.get("s_committed_ci95", [None, None])],
        "clear": _round(r["clear_winner_agreement"]),
        "tier": r.get("tier"),
        "n_models": r.get("n_models", len(rec["ratings"])),
        "n_queries": r.get("n_queries_scored", rec["config"]["n_queries"]),
    }


def select_records(
    results: list[Result], judge: str | None, generator: str | None, pins: dict[str, str]
) -> tuple[dict, dict]:
    """{task: synthetic record}, {task: original record}; ambiguity is an error, not a silent pick."""
    synth: dict[str, list[dict]] = {}
    orig: dict[str, list[dict]] = {}
    for res in results:
        rec, c = res.record, res.record["config"]
        if judge and c.get("judge_model") != judge:
            continue
        if c.get("arm") == "synthetic":
            if generator and c.get("generator_model") != generator:
                continue
            synth.setdefault(rec["task_name"], []).append(rec)
        elif c.get("arm") == "original":
            orig.setdefault(rec["task_name"], []).append(rec)

    for task, h in pins.items():
        if not any(r["config"].get("config_hash") == h for r in synth.get(task, []) + orig.get(task, [])):
            raise ExportError(f"{task}: no record with config hash {h}")
    if not synth:
        raise ExportError("no synthetic-arm record matches the filters: nothing to rank")

    def pick(group: dict[str, list[dict]], arm: str) -> dict[str, dict]:
        chosen = {}
        for task, recs in group.items():
            pinned = [r for r in recs if r["config"].get("config_hash") == pins.get(task)]
            recs = pinned or recs  # a pin names one arm's record; the other arm is left alone
            if len(recs) > 1:
                hashes = ", ".join(str(r["config"].get("config_hash")) for r in recs)
                raise ExportError(f"{task}: {len(recs)} {arm} records ({hashes}); pin one with --pin {task}=HASH")
            chosen[task] = recs[0]
        return chosen

    return pick(synth, "synthetic"), pick(orig, "original")


def build_export(
    output_folder: str | Path,
    *,
    judge: str | None = None,
    generator: str | None = None,
    pins: dict[str, str] | None = None,
    allow_missing: bool = False,
) -> dict:
    """The app's data file. Ranked corpora without a scored original-arm record are an error unless
    allow_missing, which drops them from the ranking (and says so in meta.dropped)."""
    results = load_results(output_folder).results
    if not results:
        raise ExportError(f"no records under {output_folder}")
    synth, orig = select_records(results, judge, generator, pins or {})
    reliability = {t: row for t, r in orig.items() if (row := _reliability_row(r)) is not None}
    missing = sorted(t for t in synth if t not in reliability)
    if missing and not allow_missing:
        raise ExportError(
            "ranked corpora without a reliability row: "
            + ", ".join(missing)
            + ". Run mteb_gym.reliability on their original-arm records, or pass --allow-missing to drop them."
        )
    corpora = {t: _ranking(r) for t, r in synth.items() if t not in missing}
    judges = sorted({r["config"]["judge_model"] for r in list(synth.values()) + list(orig.values())})
    revisions = sorted({str(r.get("gym_revision")) for r in list(synth.values()) + list(orig.values())})
    return {
        "meta": {
            "generated": dt.date.today().isoformat(),
            "judge": judge or (judges[0] if len(judges) == 1 else judges),
            "experiment_commit": revisions[0] if len(revisions) == 1 else revisions,
            "mteb_version": next((r.get("mteb_version") for r in synth.values()), None),
            "dropped": missing,
        },
        "corpora": corpora,
        "reliability": reliability,
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--output-folder", default="results", help="the gym's output folder (records/ inside)")
    ap.add_argument("--out", default="leaderboard/data/leaderboard_export.json")
    ap.add_argument("--judge", default=None, help="keep records from this judge model only")
    ap.add_argument("--generator", default=None, help="keep synthetic records from this generator only")
    ap.add_argument("--pin", action="append", default=[], metavar="TASK=HASH", help="pick one synthetic record")
    ap.add_argument("--allow-missing", action="store_true", help="drop ranked corpora that have no reliability row")
    args = ap.parse_args(argv)
    pins = dict(p.split("=", 1) for p in args.pin)
    export = build_export(
        args.output_folder, judge=args.judge, generator=args.generator, pins=pins, allow_missing=args.allow_missing
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(export, indent=1))
    print(f"{len(export['corpora'])} ranked corpora, {len(export['reliability'])} reliability rows -> {out}")
    if export["meta"]["dropped"]:
        print("dropped (no reliability row): " + ", ".join(export["meta"]["dropped"]))


if __name__ == "__main__":
    main()
