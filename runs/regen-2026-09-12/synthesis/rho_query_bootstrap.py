"""Query-level bootstrap of rho_both for every record.

rho_both is the Spearman correlation between a record's Bradley-Terry ratings (two-order averaged
score_a, the package fit) and the official nDCG@10. The interval a record carries, agreement.spearman_ci95,
resamples the models, not the queries, so it says how much rho depends on the roster and nothing about how
much it depends on which queries were drawn. This script adds the query-level interval.

Query bootstrap. The record's queries (qids) are resampled with replacement, every verdict row of a drawn
query is kept with both orders (the stored score_a) and counted once per draw of its query, the Bradley-Terry
fit is redone with the package function on the weighted win matrix, and Spearman against truth is taken.
Resampling queries with multiplicity is the same as duplicating their verdict rows, so the fit is the package
fit on the resampled verdict list. Reported per record: rho_both (point, all weights one, which reproduces the
record's rho bit for bit), bootstrap mean, sd, 2.5 and 97.5 percentiles.

Model bootstrap. The models are resampled with replacement the way mteb_gym.agreement.correlate does it
(same draw scheme, resamples and seed as regen_nano.py; a resample with fewer than 3 distinct models or an
undefined rho is dropped). Where the record carries agreement.spearman_ci95 the recomputed interval must
match it to 1e-9 or the script fails.

Relation to the controls. From position_bias_controls.json each record's noise share (rho_both minus mean of
C1, the random-order single-verdict refit) and both minus C3 (rho_both minus the half-queries both-order
refit) are divided by the query-bootstrap sd, so a control difference can be read against the query-level
uncertainty of the quantity it is a difference of.

Scoring conventions follow position_bias_controls.py, whose helpers load the rows and build the fit.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
for _d in (HERE, HERE.parent / "controls"):
    if (_d / "position_bias_controls.py").exists():
        sys.path.insert(0, str(_d))
        break
from position_bias_controls import (  # noqa: E402
    CI95_BOOTSTRAP, CI95_SEED, TEMPLATE_MARK, TOL, Refitter, check_against_package, pair_rows, parse_rows, spearman)
from mteb_gym.rank import BASE, SCALE, _bradley_terry  # noqa: E402

DEFAULT_TRUTH_OVERRIDE = ["MiniMax-M2.7=NFCorpus_m27"]
DEFAULT_CONTROLS = ["analysis_out/controls/position_bias_controls.json",
                    "analysis_out/NFCorpus_m27/controls/position_bias_controls.json"]


class WeightedRefitter(Refitter):
    """The control script's refitter with a per-row weight (the number of times a row's query was drawn)."""

    def win_matrix_w(self, score_a: np.ndarray, w: np.ndarray) -> np.ndarray:
        W = np.zeros((self.n, self.n))
        np.add.at(W, (self.ia, self.ib), w * score_a)
        np.add.at(W, (self.ib, self.ia), w * (1.0 - score_a))
        return W

    def ratings_w(self, score_a: np.ndarray, w: np.ndarray) -> np.ndarray:
        r = SCALE * np.log10(np.clip(_bradley_terry(self.win_matrix_w(score_a, w)), 1e-12, None))
        return r - r.mean() + BASE

    def rho_w(self, score_a: np.ndarray, w: np.ndarray) -> float:
        return spearman(self.ratings_w(score_a, w), self.names, self.truth)


def pct(x: np.ndarray) -> dict:
    return {"mean": float(np.mean(x)), "sd": float(np.std(x, ddof=1)), "p2_5": float(np.percentile(x, 2.5)),
            "p97_5": float(np.percentile(x, 97.5)), "min": float(np.min(x)), "max": float(np.max(x)),
            "width": float(np.percentile(x, 97.5) - np.percentile(x, 2.5))}


def query_bootstrap(fit: WeightedRefitter, s_both: np.ndarray, q_index: np.ndarray, n_q: int,
                    resamples: int, seed: int) -> tuple[np.ndarray, int]:
    """Resample the n_q queries with replacement; weight = how many times each row's query was drawn."""
    rng = np.random.default_rng(seed)
    draws, dropped = [], 0
    for _ in range(resamples):
        counts = np.bincount(rng.integers(0, n_q, n_q), minlength=n_q).astype(float)
        r = fit.rho_w(s_both, counts[q_index])
        if np.isnan(r):
            dropped += 1
        else:
            draws.append(r)
    return np.array(draws), dropped


def model_bootstrap(ratings: dict, truth: dict, resamples: int, seed: int) -> np.ndarray:
    """mteb_gym.agreement.correlate's model bootstrap: models in rating-dict order, resampled with
    replacement, resamples with fewer than 3 distinct models or an undefined rho dropped."""
    shared = [m for m in ratings if m in truth]
    g = np.array([ratings[m] for m in shared])
    t = np.array([truth[m] for m in shared])
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(resamples):
        idx = rng.integers(0, len(shared), len(shared))
        if len(set(idx)) < 3:
            continue
        r, _ = spearmanr(g[idx], t[idx])
        if not np.isnan(r):
            boots.append(r)
    return np.array(boots)


def truth_dir(rec_path: Path, task: str, overrides: list[str]) -> str:
    for item in overrides:
        key, _, d = item.partition("=")
        if key and key in rec_path.name:
            return d
    return task


def analyse(args: tuple) -> dict:
    rec_path, root, resamples, seed, overrides, controls = args
    t0 = time.time()
    out, an = root / "results", root / "analysis_out"
    if TEMPLATE_MARK in rec_path.name:
        raise RuntimeError(f"template-looking record name: {rec_path.name}")
    rec = json.loads(rec_path.read_text())
    task, arm = rec["task_name"], rec["config"]["arm"]
    tdir = truth_dir(rec_path, task, overrides)
    truth_text = (an / tdir / "truth.json").read_text()
    if TEMPLATE_MARK in truth_text:
        raise RuntimeError(f"template-looking text in truth for {tdir}")
    truth = json.loads(truth_text)
    roster = list(rec["config"]["models"])
    rows, n_reversed = pair_rows(out, rec, roster)
    names = sorted({m for v in rows for m in (v["model_a"], v["model_b"])})
    idx = {m: i for i, m in enumerate(names)}
    fit = WeightedRefitter(names, np.array([idx[v["model_a"]] for v in rows]),
                           np.array([idx[v["model_b"]] for v in rows]), truth)
    arr = parse_rows(rows, task)
    s_both = arr["s_both"]
    max_rating_diff = check_against_package(rows, rec, fit, s_both)

    qids = sorted({v["qid"] for v in rows})
    q_pos = {q: i for i, q in enumerate(qids)}
    q_index = np.array([q_pos[v["qid"]] for v in rows])
    ones = np.ones(len(rows))
    rho_point = fit.rho_w(s_both, ones)
    rho_unweighted = fit.rho(s_both)
    if abs(rho_point - rho_unweighted) > TOL:
        raise RuntimeError(f"{rec_path.name}: weighted point refit {rho_point!r} != refit {rho_unweighted!r}")
    rec_rho = (rec.get("agreement") or {}).get("spearman_rho")
    if rec_rho is not None and abs(rec_rho - rho_point) > TOL:
        raise RuntimeError(f"{rec_path.name}: rho_both {rho_point!r} != record {rec_rho!r}")

    qb, dropped = query_bootstrap(fit, s_both, q_index, len(qids), resamples, seed)
    q_stats = {**pct(qb), "resamples": resamples, "n_kept": int(len(qb)), "n_dropped": dropped, "seed": seed}

    ratings = {r["model"]: r["rating"] for r in rec["ratings"]}
    mb = model_bootstrap(ratings, truth, CI95_BOOTSTRAP, CI95_SEED)
    m_stats = {**pct(mb), "resamples": CI95_BOOTSTRAP, "n_kept": int(len(mb)), "seed": CI95_SEED}
    ci = (rec.get("agreement") or {}).get("spearman_ci95")
    ci = list(ci) if ci and ci[0] is not None else None
    if ci is not None:
        diff = max(abs(m_stats["p2_5"] - ci[0]), abs(m_stats["p97_5"] - ci[1]))
        if diff > TOL:
            raise RuntimeError(f"{rec_path.name}: model bootstrap [{m_stats['p2_5']!r}, {m_stats['p97_5']!r}] != record {ci!r}")
        m_stats["record_ci95"], m_stats["record_check"], m_stats["max_abs_edge_diff"] = ci, "match", diff
    else:
        m_stats["record_ci95"], m_stats["record_check"], m_stats["max_abs_edge_diff"] = None, "absent (no agreement block)", None

    c = controls.get(rec_path.name)
    if c is None:
        raise RuntimeError(f"{rec_path.name}: not present in any controls file")
    sd_q = q_stats["sd"]
    rel = {"noise_share": c["noise_share"], "both_minus_c3": c["both_minus_c3"], "bias_share": c["bias_share"],
           "c1_sd": c["c1"]["sd"], "c3_sd": c["c3"]["sd"], "c1_mean": c["c1"]["mean"], "c3_mean": c["c3"]["mean"],
           "noise_share_over_query_sd": c["noise_share"] / sd_q,
           "both_minus_c3_over_query_sd": c["both_minus_c3"] / sd_q,
           "c3_sd_over_query_sd": c["c3"]["sd"] / sd_q, "source": c["_source"]}
    if abs(c["rho_both"] - rho_point) > TOL:
        raise RuntimeError(f"{rec_path.name}: rho_both {rho_point!r} != controls {c['rho_both']!r}")
    return {
        "task": task, "arm": arm, "judge_model": rec["config"]["judge_model"], "record": rec_path.name,
        "truth_dir": tdir, "n_models": len(names), "n_models_in_truth": sum(m in truth for m in names),
        "n_queries": rec["config"]["n_queries"], "n_unique_qids": len(qids), "n_rows": len(rows),
        "n_pair_files_reversed": n_reversed, "max_abs_rating_diff": max_rating_diff,
        "rho_both": rho_point, "rho_both_record": rec_rho,
        "query": q_stats, "model": m_stats,
        "model_width_over_query_width": m_stats["width"] / q_stats["width"] if q_stats["width"] else None,
        "controls": rel, "draws": {"query": qb.tolist()}, "seconds": time.time() - t0,
    }


def load_controls(paths: list[Path], root: Path) -> dict:
    found = {}
    for p in paths:
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        src = str(p.relative_to(root)).replace("\\", "/") if p.is_relative_to(root) else p.name
        for arm in ("synthetic", "original"):
            for r in d.get(arm, []):
                found.setdefault(r["record"], {**r, "_source": src})
    return found


def group_of(r: dict) -> str:
    return f"{r['arm']} ({r['judge_model'].split('/')[-1]})"


def summarise(rows: list[dict]) -> dict:
    def mean_of(f) -> float:
        return float(np.mean([f(r) for r in rows]))
    return {
        "n": len(rows),
        "rho_both": mean_of(lambda r: r["rho_both"]),
        "query_mean": mean_of(lambda r: r["query"]["mean"]),
        "query_sd": mean_of(lambda r: r["query"]["sd"]),
        "query_width": mean_of(lambda r: r["query"]["width"]),
        "query_bias": mean_of(lambda r: r["query"]["mean"] - r["rho_both"]),
        "model_width": mean_of(lambda r: r["model"]["width"]),
        "model_sd": mean_of(lambda r: r["model"]["sd"]),
        "model_width_over_query_width": mean_of(lambda r: r["model_width_over_query_width"]),
        "noise_share": mean_of(lambda r: r["controls"]["noise_share"]),
        "both_minus_c3": mean_of(lambda r: r["controls"]["both_minus_c3"]),
        "noise_share_over_query_sd": mean_of(lambda r: r["controls"]["noise_share_over_query_sd"]),
        "both_minus_c3_over_query_sd": mean_of(lambda r: r["controls"]["both_minus_c3_over_query_sd"]),
        "abs_noise_share_over_query_sd_max": float(max(abs(r["controls"]["noise_share_over_query_sd"]) for r in rows)),
        "abs_both_minus_c3_over_query_sd_max": float(max(abs(r["controls"]["both_minus_c3_over_query_sd"]) for r in rows)),
        "n_abs_noise_share_above_query_sd": int(sum(abs(r["controls"]["noise_share_over_query_sd"]) > 1 for r in rows)),
        "n_abs_both_minus_c3_above_query_sd": int(sum(abs(r["controls"]["both_minus_c3_over_query_sd"]) > 1 for r in rows)),
        "n_model_interval_wider": int(sum(r["model"]["width"] > r["query"]["width"] for r in rows)),
        "n_query_interval_contains_point": int(sum(r["query"]["p2_5"] <= r["rho_both"] <= r["query"]["p97_5"] for r in rows)),
        "c3_sd_over_query_sd": mean_of(lambda r: r["controls"]["c3_sd_over_query_sd"]),
        "n_model_ci95_checked": int(sum(r["model"]["record_check"] == "match" for r in rows)),
        "max_query_dropped": int(max(r["query"]["n_dropped"] for r in rows)),
    }


def f3(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def band(d: dict) -> str:
    return f"{d['mean']:.3f} (sd {d['sd']:.3f}) [{d['p2_5']:.3f}, {d['p97_5']:.3f}]"


def overlap(a: dict, b: dict) -> bool:
    return bool(a["p2_5"] <= b["p97_5"] and b["p2_5"] <= a["p97_5"])


def exceed_list(rows: list[dict], key: str) -> str:
    items = [f"{r['task']} ({group_of(r)}) {f3(r['controls'][key])}" for r in rows if abs(r["controls"][key]) > 1]
    return "; ".join(items) or "none"


def ci95_sentence(s: dict) -> str:
    k, n = s["n_model_ci95_checked"], s["n"]
    if k == n:
        return f"The model bootstrap reproduces the record interval on {n} of {n} records."
    if k == 0:
        return (f"None of the {n} records carries an agreement block, so the model interval is computed here for the "
                f"first time and there is nothing to check it against.")
    return (f"The model bootstrap reproduces the record interval on the {k} of {n} records that carry one; the other "
            f"{n - k} carry no agreement block, so their model interval is computed here for the first time.")


def results_paragraph(out: dict) -> list[str]:
    groups, S = out["groups"], out["summary"]
    rows = [r for g in groups for r in groups[g]]
    parts = []
    for g, s in S.items():
        parts.append(
            f"Over the {s['n']} record{'s' if s['n'] != 1 else ''} in the {g} group the query interval (2.5 to 97.5 percentiles over "
            f"{out['settings']['resamples']} query resamples) is {f3(s['query_width'])} wide on average with a bootstrap sd of "
            f"{f3(s['query_sd'])}, against a model interval {f3(s['model_width'])} wide (sd {f3(s['model_sd'])}); the model "
            f"interval is wider on {s['n_model_interval_wider']} of {s['n']} records, by a factor of "
            f"{s['model_width_over_query_width']:.2f} on average. The bootstrap mean sits {f3(s['query_bias'])} from the point "
            f"rho_both on average and the point lies inside the query interval on {s['n_query_interval_contains_point']} of "
            f"{s['n']}. The noise share divided by the query sd averages {f3(s['noise_share_over_query_sd'])} (largest absolute "
            f"value {s['abs_noise_share_over_query_sd_max']:.2f}); it exceeds one query sd in absolute value on "
            f"{s['n_abs_noise_share_above_query_sd']} of {s['n']} records. Both minus C3 over the query sd averages "
            f"{f3(s['both_minus_c3_over_query_sd'])} (largest absolute value {s['abs_both_minus_c3_over_query_sd_max']:.2f}), "
            f"above one in absolute value on {s['n_abs_both_minus_c3_above_query_sd']} of {s['n']}. The C3 draw sd is "
            f"{s['c3_sd_over_query_sd']:.2f} query sds on average. "
            + ci95_sentence(s))
    parts.append(f"Records whose absolute noise share exceeds one query-bootstrap sd: {exceed_list(rows, 'noise_share_over_query_sd')}. "
                 f"Records whose absolute both minus C3 exceeds one query-bootstrap sd: {exceed_list(rows, 'both_minus_c3_over_query_sd')}.")
    cmp = out.get("nfcorpus_judges")
    if cmp:
        a, b = cmp["records"]
        text = (
            f"On NFCorpus the {a['judge']} record has rho_both {f3(a['rho_both'])} with query interval [{f3(a['query']['p2_5'])}, "
            f"{f3(a['query']['p97_5'])}] and model interval [{f3(a['model']['p2_5'])}, {f3(a['model']['p97_5'])}]; the "
            f"{b['judge']} record has rho_both {f3(b['rho_both'])} with query interval [{f3(b['query']['p2_5'])}, "
            f"{f3(b['query']['p97_5'])}] and model interval [{f3(b['model']['p2_5'])}, {f3(b['model']['p97_5'])}]. The query "
            f"intervals {'overlap' if cmp['query_overlap'] else 'do not overlap'}; the model intervals "
            f"{'overlap' if cmp['model_overlap'] else 'do not overlap'}. The difference in rho_both, {f3(cmp['rho_diff'])}, is "
            f"{cmp['rho_diff_over_pooled_query_sd']:.2f} pooled query sds.")
        if cmp["paired"]:
            p = cmp["paired"]
            text += (f" Both records are judged on the same {a['n_queries']} queries and the two bootstraps share their "
                     f"resampled query sets draw for draw (same seed), so the paired difference {a['judge']} minus {b['judge']} "
                     f"has bootstrap mean {f3(p['mean'])}, sd {f3(p['sd'])} and interval [{f3(p['p2_5'])}, {f3(p['p97_5'])}].")
        parts.append(text)
    return parts


def nfcorpus_judges(rows: list[dict]) -> dict | None:
    nf = [r for r in rows if r["task"] == "NFCorpus" and r["arm"] == "synthetic"]
    mm = [r for r in nf if "MiniMax" in r["judge_model"]]
    qw = [r for r in nf if "MiniMax" not in r["judge_model"]]
    if len(mm) != 1 or len(qw) != 1:
        return None
    a, b = mm[0], qw[0]
    da, db = np.array(a["draws"]["query"]), np.array(b["draws"]["query"])
    same = a["n_unique_qids"] == b["n_unique_qids"] and len(da) == len(db) and a["query"]["seed"] == b["query"]["seed"]
    pooled = float(np.sqrt((a["query"]["sd"] ** 2 + b["query"]["sd"] ** 2) / 2))
    return {"records": [{"judge": a["judge_model"].split("/")[-1], "record": a["record"], "rho_both": a["rho_both"],
                         "query": a["query"], "model": a["model"], "n_queries": a["n_queries"]},
                        {"judge": b["judge_model"].split("/")[-1], "record": b["record"], "rho_both": b["rho_both"],
                         "query": b["query"], "model": b["model"], "n_queries": b["n_queries"]}],
            "query_overlap": overlap(a["query"], b["query"]), "model_overlap": overlap(a["model"], b["model"]),
            "rho_diff": a["rho_both"] - b["rho_both"], "rho_diff_over_pooled_query_sd": (a["rho_both"] - b["rho_both"]) / pooled,
            "same_query_draws": same, "paired": pct(da - db) if same else None}


def write_md(out: dict, path: Path) -> None:
    st = out["settings"]
    lines = ["# Query bootstrap of rho_both", "", __doc__.strip(), "",
             f"Query resamples {st['resamples']}, seed {st['seed']}; model bootstrap {st['ci95_bootstrap']} resamples, seed "
             f"{st['ci95_seed']} (the record's own setting). Each interval cell is mean (sd over resamples) [2.5, 97.5 percentiles]. "
             "n_q is the number of queries in the record. noise share and both minus C3 are taken from the controls run; the "
             "columns after them divide by the query-bootstrap sd of this run.", ""]
    cols = ("| task | n_q | rho_both | query mean (sd) [2.5, 97.5] | query width | model mean (sd) [2.5, 97.5] | model width | "
            "model width over query width | noise share | noise share over query sd | both minus C3 | both minus C3 over query sd | checks |")
    for g, rows in out["groups"].items():
        n_word = "record" if len(rows) == 1 else "records"
        lines += [f"## {g} ({len(rows)} {n_word})", "", cols, "|---|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|"]
        for r in rows:
            src = r["controls"]["source"]
            chk = [f"model ci95 {r['model']['record_check']}",
                   "point = record" if r["rho_both_record"] is not None else "no record rho",
                   f"controls from {src.split('analysis_out/')[-1].rsplit('/', 1)[0] if '/' in src else src}"]
            if r["query"]["n_dropped"]:
                chk.append(f"{r['query']['n_dropped']} query resamples dropped")
            if r["n_pair_files_reversed"]:
                chk.append(f"{r['n_pair_files_reversed']} files reversed")
            c = r["controls"]
            lines.append(f"| {r['task']} | {r['n_queries']} | {f3(r['rho_both'])} | {band(r['query'])} | {f3(r['query']['width'])} | "
                         f"{band(r['model'])} | {f3(r['model']['width'])} | {r['model_width_over_query_width']:.2f} | "
                         f"{f3(c['noise_share'])} | {c['noise_share_over_query_sd']:.2f} | {f3(c['both_minus_c3'])} | "
                         f"{c['both_minus_c3_over_query_sd']:.2f} | {'; '.join(chk)} |")
        lines.append("")
    lines += ["## Group means", "",
              "| group | n | rho_both | query mean | query sd | query width | model width | model width over query width | "
              "noise share | noise share over query sd | both minus C3 | both minus C3 over query sd | records with abs noise share above one query sd |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for g, s in out["summary"].items():
        lines.append(f"| {g} | {s['n']} | {f3(s['rho_both'])} | {f3(s['query_mean'])} | {f3(s['query_sd'])} | {f3(s['query_width'])} | "
                     f"{f3(s['model_width'])} | {s['model_width_over_query_width']:.2f} | {f3(s['noise_share'])} | "
                     f"{s['noise_share_over_query_sd']:.2f} | {f3(s['both_minus_c3'])} | {s['both_minus_c3_over_query_sd']:.2f} | "
                     f"{s['n_abs_noise_share_above_query_sd']} |")
    lines += ["", "## Results", ""]
    for p in results_paragraph(out):
        lines += [p, ""]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="run root holding results/ and analysis_out/")
    ap.add_argument("--out-dir", default=None, help="default <root>/analysis_out/controls_bootstrap")
    ap.add_argument("--resamples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", default=None, help="substring filter on record file names")
    ap.add_argument("--controls", action="append", default=None,
                    help="position_bias_controls.json to relate to (repeatable); default the controls and NFCorpus_m27 files under root")
    ap.add_argument("--truth-override", action="append", default=None,
                    help="RECORD_SUBSTRING=DIR, truth.json under analysis_out/DIR for matching records; default MiniMax-M2.7=NFCorpus_m27")
    a = ap.parse_args()
    root = Path(a.root)
    out_dir = Path(a.out_dir) if a.out_dir else root / "analysis_out" / "controls_bootstrap"
    out_dir.mkdir(parents=True, exist_ok=True)
    overrides = a.truth_override if a.truth_override is not None else DEFAULT_TRUTH_OVERRIDE
    ctl_paths = [Path(p) for p in a.controls] if a.controls else [root / p for p in DEFAULT_CONTROLS]
    controls = load_controls(ctl_paths, root)
    if not controls:
        raise SystemExit("no controls found")
    recs = sorted((root / "results" / "records").glob("*.json"))
    if a.only:
        recs = [p for p in recs if a.only in p.name]
    if not recs:
        raise SystemExit("no records")
    t0 = time.time()
    jobs = [(p, root, a.resamples, a.seed, overrides, controls) for p in recs]
    with Pool(a.workers) as pool:
        results = pool.map(analyse, jobs, chunksize=1)
    groups: dict[str, list[dict]] = {}
    for r in results:
        groups.setdefault(group_of(r), []).append(r)
        print(f"{group_of(r):28s} {r['task']:28s} both={f3(r['rho_both'])} query={band(r['query'])} w={f3(r['query']['width'])} "
              f"model={band(r['model'])} w={f3(r['model']['width'])} noise/sd={r['controls']['noise_share_over_query_sd']:.2f} "
              f"(both-C3)/sd={r['controls']['both_minus_c3_over_query_sd']:.2f} ci95={r['model']['record_check']} "
              f"{r['seconds']:.0f}s", flush=True)
    order = sorted(groups, key=lambda g: (not g.startswith("synthetic"), "MiniMax" in g, g))
    out = {"groups": {g: groups[g] for g in order}, "summary": {g: summarise(groups[g]) for g in order},
           "nfcorpus_judges": nfcorpus_judges(results),
           "settings": {"resamples": a.resamples, "seed": a.seed, "workers": a.workers, "n_records": len(results),
                        "ci95_bootstrap": CI95_BOOTSTRAP, "ci95_seed": CI95_SEED, "truth_override": overrides,
                        "controls_sources": sorted({r["controls"]["source"] for r in results}),
                        "wall_seconds": time.time() - t0},
           "definitions": __doc__}
    (out_dir / "rho_query_bootstrap.json").write_text(json.dumps(out, indent=1))
    write_md(out, out_dir / "rho_query_bootstrap.md")
    print(f"wrote {out_dir / 'rho_query_bootstrap.json'} and .md in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
